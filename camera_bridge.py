"""
Camera Bridge
=============
Streams RealSense frames over TCP to a server.py that can't reach the camera
itself — e.g. server.py running in Docker on a Mac, where the container can
neither install pyrealsense2 (no linux/arm64 wheel) nor see USB devices.

Run this natively on the machine the camera is plugged into, inside the
Python environment that has pyrealsense2 (on the Mac, your venv):

    source path/to/venv/bin/activate
    python camera_bridge.py                # listens on 0.0.0.0:8765

Then start the web app as usual (`docker compose up --build`): inside the
container create_camera() finds no pyrealsense2 and automatically connects
back to this bridge at host.docker.internal:8765. Point a container on a
different machine here with CAMERA_BRIDGE=<this-host>:8765.

Wire format per frame: an 8-byte !II header with (jpeg_len, png_len), then a
JPEG color image and a lossless 16-bit PNG depth map (z16, 1 unit = 1mm) —
decoded by NetworkCamera in camera_integration.py.
"""

import socket
import struct
import time

import cv2
import numpy as np

from camera_integration import RealSenseCamera

HOST, PORT   = "0.0.0.0", 8765
JPEG_QUALITY = 85


def _stream(camera, conn):
    while True:
        color, depth_frame = camera.get_frame()
        if color is None:
            continue
        depth_array = np.asanyarray(depth_frame.get_data())
        ok_c, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        ok_d, dpng = cv2.imencode(".png", depth_array)
        if not (ok_c and ok_d):
            continue
        conn.sendall(struct.pack("!II", len(jpeg), len(dpng))
                     + jpeg.tobytes() + dpng.tobytes())


def main():
    camera = RealSenseCamera()
    while True:
        try:
            camera.start()
            break
        except RuntimeError as e:
            # Typically "No device connected" — camera unplugged, on a
            # charge-only/USB-2 cable, or grabbed by another process.
            print(f"[bridge] {e} — waiting for the RealSense (USB 3 port, "
                  "no other app using it). Retrying in 3s...")
            time.sleep(3)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[bridge] Streaming camera on {HOST}:{PORT} — Ctrl+C to stop")

    try:
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[bridge] Client connected: {addr[0]}:{addr[1]}")
            try:
                _stream(camera, conn)
            except OSError:
                print("[bridge] Client disconnected")
            finally:
                conn.close()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        camera.stop()
        print("[bridge] Done.")


if __name__ == "__main__":
    main()
