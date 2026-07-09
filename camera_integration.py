"""
RealSense Camera Interface
===========================
Wraps the Intel RealSense pipeline (aligned color + depth, with spatial /
temporal / hole-filling filters) behind a small class so other scripts
(e.g. model.py) can pull frames without touching the SDK directly.

Run this file directly for the standalone depth/measure viewer:
  Controls: Q=quit  S=save  C=colormap  Click=measure
"""

import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    # No wheel on this platform (macOS, Linux aarch64) — see requirements.txt.
    # create_camera() falls back to NetworkCamera (frames from camera_bridge.py
    # on the host) or NullCamera so the web UI still comes up.
    rs = None

DEPTH_UNITS = 0.001  # z16 format: each unit = 1mm = 0.001m


class RealSenseCamera:
    """Starts a RealSense pipeline and hands back aligned color/depth frames."""

    def __init__(self, width=640, height=480, fps=30, warmup_s=3):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed (no wheel for this platform) — "
                "use create_camera() to fall back to a placeholder feed")
        self.width = width
        self.height = height

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.align = rs.align(rs.stream.color)

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)

        self._warmup_s = warmup_s
        self._depth_frame = None  # latest filtered depth frame, for get_distance_cm()

    def start(self):
        self.pipeline.start(self.config)
        print("[RealSense] Warming up...")
        time.sleep(self._warmup_s)

    def get_frame(self):
        """Return (color_image, depth_frame) for the latest aligned pair, or (None, None) on timeout/dropped frame."""
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        self._depth_frame = depth_frame
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    def get_distance_cm(self, x, y):
        """Real-world distance in cm at pixel (x, y) of the last frame, or None if unavailable."""
        if self._depth_frame is None:
            return None
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        d = self._depth_frame.get_distance(int(x), int(y)) * 100
        return d if d > 0 else None

    def colorize_depth(self, depth_frame):
        colorized = self.colorizer.colorize(depth_frame)
        return np.asanyarray(colorized.get_data())

    def cycle_color_scheme(self):
        """Advance the depth colorizer's color scheme, return its name."""
        self._scheme = getattr(self, "_scheme", 0)
        self._scheme = (self._scheme + 1) % 6
        self.colorizer.set_option(rs.option.color_scheme, self._scheme)
        schemes = {0: "Jet", 1: "Classic", 2: "White-Black", 3: "Near-White", 4: "Cold", 5: "Warm"}
        return schemes[self._scheme]

    def stop(self):
        self.pipeline.stop()


def _text_frame(lines, width, height):
    """A black frame with the given status lines, for the no-camera fallbacks."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (30, height // 2 - 10 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


class NullCamera:
    """Placeholder for platforms without pyrealsense2 (macOS, Linux aarch64):
    serves a static frame so the web UI comes up, with no depth data."""

    def __init__(self, width=640, height=480, fps=30, warmup_s=0):
        self.width = width
        self.height = height
        self._delay = 1.0 / fps
        self._frame = _text_frame(["No RealSense camera",
                                   "pyrealsense2 unavailable on this platform"],
                                  width, height)

    def start(self):
        print("[camera] pyrealsense2 not available — serving placeholder frames")

    def get_frame(self):
        time.sleep(self._delay)  # pace the capture loop like a real camera would
        return self._frame.copy(), None

    def get_distance_cm(self, x, y):
        return None

    def colorize_depth(self, depth_frame):
        return self._frame.copy()

    def cycle_color_scheme(self):
        return "N/A"

    def stop(self):
        pass


class NetworkCamera:
    """Receives frames streamed by camera_bridge.py running on another machine
    (typically the Docker host, where the RealSense is plugged in and
    pyrealsense2 works natively). Reconnects automatically, and serves a
    status frame until the bridge is reachable.

    Wire format per frame (see camera_bridge.py): an 8-byte !II header with
    (jpeg_len, png_len), then a JPEG color image and a 16-bit PNG depth map.
    """

    def __init__(self, address="host.docker.internal:8765",
                 width=640, height=480, fps=30, warmup_s=0):
        host, _, port = address.partition(":")
        self._addr = (host, int(port) if port else 8765)
        self.width = width
        self.height = height
        self._delay = 1.0 / fps

        self._lock = threading.Lock()
        self._color = None   # latest decoded BGR frame
        self._depth = None   # latest decoded uint16 depth array (mm)
        self._running = True
        self._placeholder = _text_frame(
            ["Waiting for camera bridge at",
             f"{self._addr[0]}:{self._addr[1]}",
             "Run `python camera_bridge.py` on the camera host"],
            width, height)

    def start(self):
        print(f"[camera] Connecting to camera bridge at {self._addr[0]}:{self._addr[1]}")
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _recv_loop(self):
        while self._running:
            try:
                with socket.create_connection(self._addr, timeout=5) as sock:
                    sock.settimeout(10)
                    print("[camera] Bridge connected")
                    while self._running:
                        jpeg_len, png_len = struct.unpack("!II", self._recv_exact(sock, 8))
                        jpeg = self._recv_exact(sock, jpeg_len)
                        dpng = self._recv_exact(sock, png_len)
                        color = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        depth = cv2.imdecode(np.frombuffer(dpng, np.uint8), cv2.IMREAD_UNCHANGED)
                        with self._lock:
                            self._color, self._depth = color, depth
            except OSError:
                with self._lock:
                    self._color = self._depth = None
                time.sleep(2)  # bridge not up (yet) — keep retrying

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("bridge closed the connection")
            buf += chunk
        return buf

    def get_frame(self):
        time.sleep(self._delay)  # pace the capture loop like a real camera would
        with self._lock:
            if self._color is None:
                return self._placeholder.copy(), None
            # copy: the tracker annotates the frame in place
            return self._color.copy(), self._depth

    def get_distance_cm(self, x, y):
        with self._lock:
            depth = self._depth
        if depth is None or not (0 <= x < self.width and 0 <= y < self.height):
            return None
        d = float(depth[int(y), int(x)]) * DEPTH_UNITS * 100
        return d if d > 0 else None

    def colorize_depth(self, depth_frame):
        with self._lock:
            depth = self._depth
        if depth is None:
            return self._placeholder.copy()
        # Match the RealSense colorizer's default range (0-4m) roughly.
        scaled = np.clip(depth / 4000.0 * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    def cycle_color_scheme(self):
        return "N/A"

    def stop(self):
        self._running = False


def create_camera(**kwargs):
    """Pick the frame source for this environment:
    - CAMERA_BRIDGE=host[:port] env var  -> NetworkCamera at that address
    - pyrealsense2 importable            -> RealSenseCamera (local USB camera)
    - inside Docker without pyrealsense2 -> NetworkCamera at the Docker host
      (start camera_bridge.py natively on the host to feed it)
    - otherwise                          -> NullCamera placeholder
    """
    bridge = os.environ.get("CAMERA_BRIDGE")
    if bridge:
        return NetworkCamera(bridge, **kwargs)
    if rs is not None:
        return RealSenseCamera(**kwargs)
    if os.path.exists("/.dockerenv"):
        return NetworkCamera(**kwargs)
    return NullCamera(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone depth/measure viewer (for debugging the camera on its own)
# ─────────────────────────────────────────────────────────────────────────────
def _run_standalone_viewer():
    camera = RealSenseCamera()
    camera.start()

    click_point = None

    def on_mouse(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x % camera.width, y)

    cv2.namedWindow("RealSense Viewer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("RealSense Viewer", on_mouse)
    print("Controls: Q=quit  S=save  C=colormap  Click=measure")

    frame_count = 0

    def put(img, text, y, color=(255, 255, 255)):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    try:
        while True:
            color_image, depth_frame = camera.get_frame()
            if color_image is None:
                continue

            depth_array = np.asanyarray(depth_frame.get_data())
            depth_colormap = camera.colorize_depth(depth_frame)

            valid = depth_array[depth_array > 0]
            coverage = 100 * len(valid) / depth_array.size if len(valid) > 0 else 0
            avg_d = valid.mean() * DEPTH_UNITS * 100 if len(valid) > 0 else 0
            min_d = valid.min() * DEPTH_UNITS * 100 if len(valid) > 0 else 0

            cx, cy = camera.width // 2, camera.height // 2
            center_dist = depth_frame.get_distance(cx, cy) * 100
            for img in [color_image, depth_colormap]:
                cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

            if click_point:
                px, py = click_point
                if 0 <= px < camera.width and 0 <= py < camera.height:
                    d = depth_frame.get_distance(px, py) * 100
                    for img in [color_image, depth_colormap]:
                        cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
                        label = f"{d:.1f} cm" if d > 0 else "no data"
                        cv2.putText(img, label, (px + 10, py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            put(color_image, f"Center: {center_dist:.1f} cm", 30, (0, 255, 0))
            put(color_image, f"Coverage: {coverage:.1f}%", 60)
            put(color_image, f"Avg depth: {avg_d:.1f} cm", 90)
            put(color_image, f"Closest: {min_d:.1f} cm", 120)
            put(color_image, f"Frame: {frame_count}", 150)
            put(color_image, "S=save  C=colormap  Q=quit", 470)
            put(depth_colormap, "DEPTH MAP", 30, (255, 255, 0))
            put(depth_colormap, "Click anywhere to measure", 470)

            if len(valid) > 0:
                hist_h = 60
                hist_img = np.zeros((hist_h, camera.width, 3), dtype=np.uint8)
                hist, _ = np.histogram(valid * DEPTH_UNITS, bins=64, range=(0, 4))
                hist_norm = (hist / hist.max() * (hist_h - 5)).astype(int)
                for i, val in enumerate(hist_norm):
                    x0 = int(i * camera.width / 64)
                    x1 = int((i + 1) * camera.width / 64)
                    cv2.rectangle(hist_img, (x0, hist_h - val), (x1, hist_h), (100, 200, 255), -1)
                cv2.putText(hist_img, "0m", (2, hist_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(hist_img, "4m", (615, hist_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                depth_colormap = np.vstack([depth_colormap, hist_img])
                color_image = np.vstack([color_image, np.zeros((hist_h, camera.width, 3), dtype=np.uint8)])

            combined = np.hstack([color_image, depth_colormap])
            cv2.imshow("RealSense Viewer", combined)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"capture_{int(time.time())}.png"
                cv2.imwrite(fname, combined)
                print(f"Saved: {fname}")
            elif key == ord('c'):
                print(f"Color scheme: {camera.cycle_color_scheme()}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    _run_standalone_viewer()
