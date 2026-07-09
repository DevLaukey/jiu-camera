#!/usr/bin/env bash
# One-command launcher for a Mac (or any host where Docker can't see USB):
# starts camera_bridge.py natively in the venv that has pyrealsense2, then
# the Dockerized web app. Ctrl+C stops both.
#
#   ./run-mac.sh [path/to/venv]     # venv defaults to ./venv
#
# Why two processes: pyrealsense2 has no Linux arm64 wheel and Docker on
# macOS has no USB passthrough, so the camera must be read natively and
# streamed into the container (see camera_bridge.py).
set -euo pipefail
cd "$(dirname "$0")"

VENV="${1:-venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "No venv at '$VENV' — pass the path of the venv that has pyrealsense2:" >&2
    echo "    ./run-mac.sh /path/to/venv" >&2
    exit 1
fi
if ! "$VENV/bin/python" -c "import pyrealsense2" 2>/dev/null; then
    echo "'$VENV' has no working pyrealsense2 — point this script at the venv" >&2
    echo "where the standalone viewer (camera_integration.py) runs." >&2
    exit 1
fi

"$VENV/bin/python" camera_bridge.py &
BRIDGE_PID=$!
trap 'kill "$BRIDGE_PID" 2>/dev/null; docker compose down' EXIT

echo "[run-mac] Bridge started (pid $BRIDGE_PID); starting web app..."
docker compose up --build
