"""
Local runner for the Fighter Boundary Tracker
================================================
Opens a native OpenCV window driven by the RealSense camera. Use this when
you have a monitor attached to the machine running the camera.

For headless operation (e.g. a Jetson controlled remotely over SSH from a
MacBook, with no monitor attached) run server.py instead and control the
tracker from a browser — see tracker.py for the shared control scheme.

  python model.py [--pose]

KEYBOARD  (see tracker.py docstring for the full control scheme)
  Q  quit   (local runner only — server.py has no equivalent; stop it
             with Ctrl+C on the host instead)
"""

import argparse

import cv2

from camera_integration import RealSenseCamera
from tracker import FighterTracker, WINDOW_NAME

_MOUSE_EVENT_MAP = {
    cv2.EVENT_LBUTTONDOWN: "down",
    cv2.EVENT_MOUSEMOVE:   "move",
    cv2.EVENT_LBUTTONUP:   "up",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", action="store_true")
    args = parser.parse_args()

    camera = RealSenseCamera()
    camera.start()

    tracker = FighterTracker(use_pose=args.pose)
    tracker.load_model()

    def on_mouse(event, x, y, flags, param):
        mapped = _MOUSE_EVENT_MAP.get(event)
        if mapped:
            tracker.on_mouse_event(mapped, x, y)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print("[INFO] Press H for help, Q to quit.")

    try:
        while True:
            frame, depth_frame = camera.get_frame()
            if frame is None:
                continue

            annotated = tracker.process_frame(frame, camera)
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:                # ESC
                tracker.on_key("esc")
            elif key != 255:               # 255 == no key pressed this tick
                tracker.on_key(chr(key))
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        tracker.shutdown()


if __name__ == "__main__":
    main()
