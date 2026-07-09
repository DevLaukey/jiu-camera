# Setup Guide — Fighter Boundary Tracker on Jetson Nano

This walks through going from an unflashed Jetson Nano to a working system:
camera + YOLO tracking running headless on the Nano, controlled from a
browser on your MacBook.

**Pieces involved** (for reference — see each file's docstring for detail):

| File | Runs on | Purpose |
|---|---|---|
| `camera_integration.py` | Jetson | `RealSenseCamera` — wraps the RealSense SDK |
| `tracker.py` | Jetson | All tracking/role/boundary logic, no display dependency |
| `server.py` | Jetson | Web UI: MJPEG stream + WebSocket control (**use this**) |
| `model.py` | Jetson (with monitor) | Local OpenCV-window version, for a Nano with a screen attached |

**Hardware needed**
- Jetson Nano Developer Kit (assumed: original 4GB board, not Orin Nano — the
  steps differ if you have an Orin Nano; check `cat /etc/nv_tegra_release`
  after flashing and see the note in Part 5 if versions look off)
- microSD card, 64GB+ (32GB is the bare minimum; YOLO + PyTorch eat space)
- 5V/4A barrel-jack power supply (the micro-USB port cannot power the Nano
  reliably once a camera + Wi-Fi + GPU inference are all running)
- Intel RealSense camera (D415/D435/D455) — **must** go in a USB **3.0** port
  (the blue ones) or depth+color streaming will fail or crawl
- MacBook on the same network as the Nano, or an Ethernet cable directly
  between them

---

## Part 0 — Alternative: running directly on a MacBook (M4), no Jetson

Skip all of Parts 1–4 below if you'd rather run everything locally on the
Mac with the RealSense camera plugged straight into it (no Jetson, no SSH,
no network hop). The tracking code itself has no Jetson-specific
dependency — it's just OpenCV + PyTorch/ultralytics + pyrealsense2 — but
`pyrealsense2` has **no prebuilt macOS wheel at all** (Intel dropped
official macOS builds), so it must be built from source, same idea as the
Jetson build but using Homebrew instead of apt:

```bash
brew install librealsense cmake python@3.11
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=true \
         -DPYTHON_EXECUTABLE=$(which python3.11) \
         -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
sudo make install
```

This produces `pyrealsense2*.so` under `build/Release` (or wherever
`CMAKE_INSTALL_PREFIX` points). Copy or symlink it into your venv's
site-packages so `import pyrealsense2` resolves there:

```bash
python3.11 -m venv ~/tracker-venv
source ~/tracker-venv/bin/activate
pip install --upgrade pip
ln -s "$(pwd)/Release/pyrealsense2*.so" \
      "$(python -c 'import site; print(site.getsitepackages()[0])')/"
```

Then install the rest of the requirements — skip the `pyrealsense2` line
in `requirements.txt` since it's already installed above (the
`; platform_system != "Darwin"` marker on that line does this
automatically for you if you just run `pip install -r requirements.txt`):

```bash
cd /path/to/camera
pip install -r requirements.txt
```

Verify the build:

```bash
python3 -c "import pyrealsense2 as rs; print(rs.pipeline())"
```

From here everything runs exactly like the "with monitor" case — plug the
RealSense into a USB 3.0 (blue, or the right-side Thunderbolt/USB-C ports
on Apple Silicon Macs) port and run either:

```bash
python model.py          # native OpenCV window, since you have a screen
# or
python server.py         # browser UI at http://localhost:8000
```

No SSH, no separate host — the Mac is both the camera host and the
display. Continue reading below only if you actually want the
Jetson-hosted/headless setup instead.

## Part 1 — Flash JetPack

1. Download **balenaEtcher** on the Mac: https://etcher.balena.io
2. Download the **JetPack 4.6.x SD card image** for Jetson Nano from
   NVIDIA's Jetson Nano developer kit page (search "Jetson Nano SD Card
   Image" on NVIDIA's site — grab the `.zip`, don't unzip it).
3. In Etcher: select the downloaded `.zip`, select the microSD card, flash.
   Takes ~15–20 minutes.
4. Insert the card into the Nano, connect a monitor + keyboard for this
   *one-time* setup step (you can't do the initial OEM config over SSH),
   plug in the barrel-jack power supply, and boot.
5. Walk through Ubuntu's first-boot wizard (accept license, set timezone,
   create a username/password — **remember these**, you'll SSH in with
   them). Let it reboot when done.

## Part 2 — Enable SSH access and go headless

The Nano ships with SSH already enabled by default on JetPack. From the
Nano's desktop (still with monitor/keyboard attached):

```bash
hostname -I          # note the IP address, e.g. 192.168.1.42
```

From the MacBook, confirm you can reach it:

```bash
ssh <your-username>@192.168.1.42
```

If that works, you're done with the monitor/keyboard — unplug them and do
everything else over SSH.

**Optional but recommended — reach it by name instead of IP:**
Ubuntu ships with `avahi-daemon`, so `<hostname>.local` usually already
resolves. Check the Nano's hostname with `hostname`, then from the Mac:

```bash
ssh <your-username>@<hostname>.local
```

**Free up RAM (the Nano only has 4GB and YOLO + camera will feel it):**
since you're running headless, stop the desktop environment from
auto-starting:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

(To get the desktop back later if you ever need a monitor again:
`sudo systemctl set-default graphical.target`.)

**Add swap** — 4GB RAM is tight for YOLO inference + build steps later.
NVIDIA ships a helper script for this on JetPack images:

```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
echo '/mnt/4GB.swap swap swap defaults 0 0' | sudo tee -a /etc/fstab
free -h   # confirm swap shows up
```

## Part 3 — Get the project onto the Nano

From your MacBook (replace the path/host as needed):

```bash
scp -r "c:/Users/hp/Videos/SAM/camera" <your-username>@<hostname>.local:~/fighter-tracker
```

(Or, if the project is in git, just `git clone` it directly on the Nano —
either works.)

SSH into the Nano and go to that directory for everything below:

```bash
ssh <your-username>@<hostname>.local
cd ~/fighter-tracker
```

## Part 4 — System dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential cmake git \
    libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
```

### Intel RealSense SDK (`pyrealsense2`)

This is the single trickiest part of this whole setup — `pip install
pyrealsense2` does **not** work on Jetson's ARM CPU (the PyPI wheels are
x86_64-only), so librealsense has to be built from source. Budget
**1–2 hours** for the Nano to compile it.

The community-maintained JetsonHacks installer automates this correctly
(handles the udev rules and kernel-module quirks that trip up the official
librealsense build instructions on Jetson):

```bash
git clone https://github.com/JetsonHacksNano/installRealSenseSDK.git
cd installRealSenseSDK
./buildLibrealsense.sh    # go make coffee, this takes a while
cd ..
```

Verify it built and the camera is detected (plug the RealSense into a
**USB 3.0 (blue)** port first):

```bash
python3 -c "import pyrealsense2 as rs; print(rs.pipeline())"
rs-enumerate-devices          # should list your D4xx camera
```

If `rs-enumerate-devices` shows nothing: reseat the USB cable in a blue
port, try a different USB-C/USB-A cable (some are power-only), and re-run
`rs-enumerate-devices`.

### PyTorch + Ultralytics (YOLO)

JetPack 4.6 ships **Python 3.6**, and current `ultralytics` requires Python
3.8+. Don't fight this with pip on system Python — it will spiral into
dependency hell. Two working options, pick one:

**Option A (recommended) — NVIDIA's `jetson-containers` (Docker)**
This gives you a container with a matching PyTorch/Python/CUDA already
built and tested for the Nano, and you `pip install` just your extra deps
(fastapi etc.) inside it.

```bash
git clone https://github.com/dusty-nv/jetson-containers.git
cd jetson-containers
./install.sh
./run.sh --volume ~/fighter-tracker:/fighter-tracker \
         --workdir /fighter-tracker \
         $(./autotag ultralytics)
```

That drops you into a shell inside the container with the project mounted
at `/fighter-tracker` and PyTorch/ultralytics already working. From inside
the container, jump to **Part 5** below (installing the web-UI deps), then
**Part 6** to run it — everything else is identical.

**Option B — bare-metal Python 3.8 venv**
More fragile (community wheels, may need adjusting versions), but avoids
Docker. Use if Option A isn't viable for you.

```bash
sudo apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv ~/tracker-venv
source ~/tracker-venv/bin/activate
pip install --upgrade pip
# NVIDIA forum PyTorch wheel for JetPack 4.6 / Python 3.8 (aarch64):
# search "PyTorch for Jetson" on the NVIDIA developer forums for the current
# download link matching your JetPack version, then:
pip install <torch-wheel-url>.whl
pip install ultralytics
```

## Part 5 — Web UI dependencies

Inside whichever environment you set up in Part 4 (the Docker container's
shell, or the activated `tracker-venv`):

```bash
pip install fastapi "uvicorn[standard]" jinja2 opencv-python numpy
```

(`pyrealsense2` was already built/installed in Part 4 — don't `pip install`
it, that'll pull the incompatible x86 wheel and shadow the real one.)

## Part 6 — First run

Sanity-check the camera on its own first (no YOLO, no web server) — this
confirms `pyrealsense2` and the camera hardware are working before adding
more moving parts. Since the Nano is headless, use this quick script
instead of `camera_integration.py`'s `cv2.imshow`-based standalone viewer
(which needs a display):

```bash
python3 -c "
from camera_integration import RealSenseCamera
import cv2
cam = RealSenseCamera()
cam.start()
frame, depth = cam.get_frame()
cv2.imwrite('test_frame.jpg', frame)
print('center distance (cm):', cam.get_distance_cm(320, 240))
cam.stop()
"
```

Copy `test_frame.jpg` back to your Mac (`scp` it) and open it — you should
see a color frame from the camera. If `get_distance_cm` printed a sane
number (not `None`), depth is working too.

Now run the actual server:

```bash
python3 server.py
```

First launch will download the YOLO weights (`yolov8n.pt`, a few MB) —
give it a minute. Once you see Uvicorn's `Application startup complete`,
it's live.

## Part 7 — Connect from the MacBook

**Option A — same network, direct:**
Open a browser on the Mac and go to:

```
http://<hostname>.local:8000
```

(or `http://<nano-ip>:8000` if `.local` resolution isn't working for you).

**Option B — tunnel over SSH** (useful if you don't want port 8000 exposed
on the network, or you're on separate networks and SSH-ing through a
jump host):

```bash
ssh -L 8000:localhost:8000 <your-username>@<hostname>.local
```

Leave that SSH session open, then browse to `http://localhost:8000` on the
Mac — traffic is tunneled through SSH instead of hitting the Nano's port
directly.

You should see the live annotated feed with the button panel from
`templates/index.html`. Try: click **Fighter 1**, then click a detected
person in the video — you should see a log line and the box turn into
"Fighter 1". Draw a boundary with **Draw** → click a few points in the
video → **Finish**.

To stop the server: `Ctrl+C` in the SSH session running `server.py`. (No
in-app "quit" — that's intentional, see `model.py`'s docstring; killing the
process is the shutdown path, and it saves the boundary on the way out.)

## Part 8 — Performance tuning (if it feels laggy)

- **Max out the clocks:** `sudo nvpmodel -m 0 && sudo jetson_clocks` (locks
  CPU/GPU to max frequency; the Nano throttles by default).
- **Lower resolution/fps:** `RealSenseCamera(width=424, height=240, fps=15)`
  in `server.py` where `camera = RealSenseCamera()` is constructed — the
  Nano's Maxwell GPU is small, 640×480@30 running YOLO concurrently is a
  lot to ask.
- **Drop stream JPEG quality/rate:** `JPEG_QUALITY` and `STREAM_FPS` at the
  top of `server.py` — quality 60–70 and 10–15fps is usually indistinguishable
  in a browser but meaningfully cheaper to encode.
- **Export YOLO to TensorRT** for a real speedup (2–4x typical on Nano):
  ```bash
  yolo export model=yolov8n.pt format=engine device=0
  ```
  then point `tracker.py`'s `load_model()` at the resulting `.engine` file
  instead of the `.pt` — bigger change, only worth it once the basics are
  working smoothly.

## Part 9 — Alternative: running in Docker (Linux hosts)

Instead of Parts 4–6, a Linux host (x86 box, or a Jetson if you accept
CPU-only inference — see the note below) can run the whole thing in a
container:

```bash
docker compose up --build     # then open http://<host>:8000
```

What the image handles for you:

- `pyrealsense2` installs from its Linux pip wheel (x86_64 and aarch64 both
  exist), so none of the from-source librealsense build in Part 4 is needed.
- YOLO weights (`yolov8n.pt` and `yolov8n-pose.pt`) are baked in at build
  time, so the container starts without network access.
- Torch is the CPU-only build to keep the image small (~2 GB instead of ~7).

Caveats:

- **Linux hosts only.** On Windows/macOS, Docker runs inside a VM that can't
  see USB devices, so the camera is unreachable — run natively there instead.
- **The camera is passed through** via `/dev/bus/usb` plus device cgroup
  rules in `docker-compose.yml`. If the container can't find the camera,
  switch to the `privileged: true` fallback noted in that file.
- **No GPU inference as-is.** On a Jetson, GPU-accelerated containers need
  NVIDIA's L4T base image and `nvidia-container-runtime` instead of the
  `python:3.11-slim` base — a separate exercise; the native install
  (Parts 4–6) is the better-trodden path there.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `rs-enumerate-devices` shows nothing | Camera in a USB 2.0 (black) port, or a charge-only cable |
| `ImportError: pyrealsense2` inside venv/container | Built it in a different environment than the one you're running `server.py` from — rebuild or reinstall in the right one |
| Nano randomly reboots under load | Power supply isn't actually 5V/4A via barrel jack, or J48 jumper (enables barrel-jack power) isn't set — check the Nano's jumper pins |
| Browser shows a broken image / stream never loads | `server.py` isn't running, or a firewall is blocking port 8000 — check `sudo ufw status` on the Nano |
| Everything works but is slow / choppy | See Part 8 |
| `ultralytics` pip install fails with Python version errors | You're on system Python 3.6 — use Option A or B from Part 4, don't fight it |
