"""
Web UI for the Fighter Boundary Tracker
==========================================
Streams the annotated RealSense feed as MJPEG and accepts click/drag/key
control messages over a WebSocket, so the tracker can run headless on a
camera host (e.g. a Jetson Nano) and be operated from a browser on another
machine (e.g. a MacBook over the LAN or an SSH port-forward) — no X11
forwarding or VNC required.

    python server.py                      # http://<host>:8000
    ssh -L 8000:localhost:8000 jetson      # then open localhost:8000 on the Mac

See tracker.py for the control scheme (buttons mirror it 1:1; keyboard
shortcuts also work while the page has focus).
"""

import asyncio
import threading

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from camera_integration import RealSenseCamera
from tracker import FighterTracker

STREAM_FPS    = 20
JPEG_QUALITY  = 80

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

camera  = RealSenseCamera()
tracker = FighterTracker()

# Guards tracker's mutable state, which is written from both the capture
# thread (process_frame) and the asyncio event loop (control messages).
_tracker_lock = threading.Lock()

_frame_lock  = threading.Lock()
_latest_jpeg = None
_running     = True


def _capture_loop():
    global _latest_jpeg
    camera.start()
    tracker.load_model()

    while _running:
        frame, depth_frame = camera.get_frame()
        if frame is None:
            continue

        with _tracker_lock:
            annotated = tracker.process_frame(frame, camera)

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with _frame_lock:
                _latest_jpeg = buf.tobytes()


@app.on_event("startup")
def _startup():
    threading.Thread(target=_capture_loop, daemon=True).start()


@app.on_event("shutdown")
def _shutdown():
    global _running
    _running = False
    camera.stop()
    with _tracker_lock:
        tracker.shutdown()


async def _mjpeg_generator():
    boundary = b"--frame\r\n"
    while True:
        with _frame_lock:
            jpeg = _latest_jpeg
        if jpeg is not None:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(1 / STREAM_FPS)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _state_snapshot():
    return {
        "drawing_mode":   tracker.drawing_mode,
        "use_pose":       tracker.use_pose,
        "show_help":      tracker.show_help,
        "pending_role":   tracker.pending_role,
        "penalty_counts": tracker.penalty_counts,
        "role_to_tid":    tracker.role_to_tid,
    }


@app.websocket("/ws/control")
async def ws_control(ws: WebSocket):
    await ws.accept()
    # Send current state immediately so a freshly-connected client isn't blank.
    with _tracker_lock:
        await ws.send_json({"logs": [], "state": _state_snapshot()})
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            with _tracker_lock:
                if msg_type == "mouse":
                    tracker.on_mouse_event(data["event"], int(data["x"]), int(data["y"]))
                elif msg_type == "key":
                    tracker.on_key(str(data["key"]))
                logs  = tracker.pop_logs()
                state = _state_snapshot()

            await ws.send_json({"logs": logs, "state": state})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
