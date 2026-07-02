import pyrealsense2 as rs
import numpy as np
import cv2
import time

# --- Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0)

click_point = None

def on_mouse(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        px = x % 640
        py = y
        click_point = (px, py)

cv2.namedWindow("RealSense Viewer", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("RealSense Viewer", on_mouse)

print("Warming up...")
time.sleep(3)
print("Controls: Q=quit  S=save  C=colormap  Click=measure")

color_scheme = 0
frame_count = 0
DEPTH_UNITS = 0.001  # z16 format: each unit = 1mm = 0.001m

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Apply filters then cast back to depth_frame
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()  # ← the fix

        depth_array = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        colorized = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(colorized.get_data())

        # --- Stats ---
        valid = depth_array[depth_array > 0]
        coverage = 100 * len(valid) / depth_array.size if len(valid) > 0 else 0
        avg_d  = valid.mean()  * DEPTH_UNITS * 100 if len(valid) > 0 else 0
        min_d  = valid.min()   * DEPTH_UNITS * 100 if len(valid) > 0 else 0

        # --- Center crosshair ---
        cx, cy = 320, 240
        center_dist = depth_frame.get_distance(cx, cy) * 100
        for img in [color_image, depth_colormap]:
            cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

        # --- Click to measure ---
        if click_point:
            px, py = click_point
            if 0 <= px < 640 and 0 <= py < 480:
                d = depth_frame.get_distance(px, py) * 100
                for img in [color_image, depth_colormap]:
                    cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
                    label = f"{d:.1f} cm" if d > 0 else "no data"
                    cv2.putText(img, label, (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Text overlay ---
        def put(img, text, y, color=(255, 255, 255)):
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        put(color_image, f"Center: {center_dist:.1f} cm",  30, (0, 255, 0))
        put(color_image, f"Coverage: {coverage:.1f}%",     60)
        put(color_image, f"Avg depth: {avg_d:.1f} cm",     90)
        put(color_image, f"Closest: {min_d:.1f} cm",      120)
        put(color_image, f"Frame: {frame_count}",          150)
        put(color_image, "S=save  C=colormap  Q=quit",    470)
        put(depth_colormap, "DEPTH MAP",                    30, (255, 255, 0))
        put(depth_colormap, "Click anywhere to measure",   470)

        # --- Histogram ---
        if len(valid) > 0:
            hist_h = 60
            hist_img = np.zeros((hist_h, 640, 3), dtype=np.uint8)
            hist, _ = np.histogram(valid * DEPTH_UNITS, bins=64, range=(0, 4))
            hist_norm = (hist / hist.max() * (hist_h - 5)).astype(int)
            for i, val in enumerate(hist_norm):
                x0 = int(i * 640 / 64)
                x1 = int((i + 1) * 640 / 64)
                cv2.rectangle(hist_img, (x0, hist_h - val), (x1, hist_h), (100, 200, 255), -1)
            cv2.putText(hist_img, "0m", (2, hist_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(hist_img, "4m", (615, hist_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            depth_colormap = np.vstack([depth_colormap, hist_img])
            color_image    = np.vstack([color_image, np.zeros((hist_h, 640, 3), dtype=np.uint8)])

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
            color_scheme = (color_scheme + 1) % 6
            colorizer.set_option(rs.option.color_scheme, color_scheme)
            schemes = {0:"Jet", 1:"Classic", 2:"White-Black", 3:"Near-White", 4:"Cold", 5:"Warm"}
            print(f"Color scheme: {schemes[color_scheme]}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")