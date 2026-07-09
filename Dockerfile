# Fighter Boundary Tracker — web UI (server.py)
#
#   docker compose up --build          # then open http://<host>:8000
#
# Camera access depends on the host (see docker-compose.yml):
#   - Linux x86_64: plug the RealSense into a USB 3.0 port; compose passes it
#     through and pyrealsense2 talks to it directly.
#   - Mac/Windows: Docker can't see USB, so run `python camera_bridge.py`
#     natively (in the Python env that has pyrealsense2) and the container
#     streams frames from it via host.docker.internal:8765 automatically.

FROM python:3.11-slim

# libgl1 + libglib2.0-0: required by opencv-python's cv2 import.
# libusb-1.0-0 + libudev1: required by the bundled librealsense in the
# pyrealsense2 wheel to enumerate and stream the USB camera.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libusb-1.0-0 \
        libudev1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first so ultralytics doesn't pull the CUDA build
# (~5 GB of GPU libraries that do nothing without an NVIDIA runtime). For GPU
# inference (e.g. Jetson with nvidia-container-runtime), drop these two lines
# and use the vendor's torch base image instead.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the YOLO weights into the image so first startup doesn't depend on
# network access. YOLO() downloads into the working directory when missing.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"

COPY camera_integration.py tracker.py server.py ./
COPY static/ static/
COPY templates/ templates/

EXPOSE 8000

CMD ["python", "server.py"]
