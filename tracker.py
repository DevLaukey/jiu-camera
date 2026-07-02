"""
Fighter Boundary Tracker — core logic
=======================================
All the state and behaviour behind the "click a person to assign them as
Fighter 1 / Fighter 2 / Referee, track a mat boundary, penalise them for
stepping out" feature, wrapped in a class with no dependency on how frames
are displayed or how input arrives. Two front-ends drive it:

  model.py   — local OpenCV window (cv2.imshow + mouse callback)
  server.py  — browser-based UI (MJPEG stream + WebSocket control),
               for headless boards (e.g. a Jetson) controlled remotely.

CONTROLS  (same meaning from either front-end)
  D        start drawing mat boundary (click corners)
  F        finish / close boundary
  C        clear boundary
  S        save boundary → saved_points.json
  P        toggle pose-mode (ankle) vs bbox-centre tracking
  1/2/R    enter "assign" mode: next click assigns Fighter1/2/Referee
  ESC      cancel assignment mode
  X        clear ALL role assignments
  H        toggle help overlay

MOUSE (normal mode)
  Left-click a person box  → assign current pending role
  Left-click+drag corner   → reposition mat corner
"""

import json
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
OUTSIDE_THRESHOLD  = 30        # consecutive frames outside → PENALTY
DRAG_RADIUS        = 14        # px to grab a polygon corner
REID_HISTORY       = 8         # appearance histograms kept per role
REID_THRESHOLD     = 0.45      # Bhattacharyya distance to accept re-id
SAVED_POINTS_FILE  = "saved_points.json"
WINDOW_NAME        = "Fighter Tracker  [H=help]"

LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

# Role definitions  →  (display_name, BGR_color)
ROLES = {
    "F1":  ("Fighter 1",  (0,   200, 255)),   # yellow-ish
    "F2":  ("Fighter 2",  (255, 100,  0  )),   # orange-blue
    "REF": ("Referee",    (180, 180, 180)),    # grey
}
PENALTY_ROLES = {"F1", "F2"}   # only fighters get penalised


# ─────────────────────────────────────────────
#  Stateless helpers
# ─────────────────────────────────────────────
def load_polygon(path: str):
    try:
        with open(path) as f:
            pts = json.load(f)
        return pts
    except FileNotFoundError:
        return []


def save_polygon(path: str, pts: list):
    with open(path, "w") as f:
        json.dump(pts, f)


def poly_array(pts):
    """Return pts as np.int32 array shaped for OpenCV."""
    return np.array(pts, dtype=np.int32)


def nearest_corner(pts, x, y):
    """Return index of nearest corner within DRAG_RADIUS, else None."""
    for i, (px, py) in enumerate(pts):
        if abs(px - x) <= DRAG_RADIUS and abs(py - y) <= DRAG_RADIUS:
            return i
    return None


def extract_histogram(frame, x1, y1, x2, y2):
    """HSV histogram of the torso region (centre 60% of box)."""
    h_img, w_img = frame.shape[:2]
    x1c = max(0, x1 + (x2 - x1) // 5)
    x2c = min(w_img, x2 - (x2 - x1) // 5)
    y1c = max(0, y1 + (y2 - y1) // 4)
    y2c = min(h_img, y2 - (y2 - y1) // 4)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def hist_distance(h1, h2):
    """Bhattacharyya distance (lower = more similar)."""
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


def point_inside(poly_arr, pt):
    """Returns True if pt is inside (or on) the polygon."""
    r = cv2.pointPolygonTest(poly_arr, (float(pt[0]), float(pt[1])), False)
    return r >= 0


def check_outside(poly_arr, use_pose, kp, x1, y1, x2, y2):
    """Return (outside_now, debug_points).
    debug_points: list of ((x,y), inside_bool) for drawing.
    """
    if use_pose and kp is not None:
        la = tuple(map(int, kp[LEFT_ANKLE]))
        ra = tuple(map(int, kp[RIGHT_ANKLE]))
        la_in = point_inside(poly_arr, la)
        ra_in = point_inside(poly_arr, ra)
        return (not la_in) and (not ra_in), [(la, la_in), (ra, ra_in)]
    else:
        cx  = (x1 + x2) // 2
        pt  = (cx, y2)
        ins = point_inside(poly_arr, pt)
        return not ins, [(pt, ins)]


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_polygon(frame, pts, drawing):
    if len(pts) < 2:
        for p in pts:
            cv2.circle(frame, tuple(p), 5, (0, 255, 0), -1)
        return
    arr = poly_array(pts)
    if drawing:
        cv2.polylines(frame, [arr], False, (0, 200, 255), 2)
        for p in pts:
            cv2.circle(frame, tuple(p), 5, (0, 200, 255), -1)
    else:
        cv2.polylines(frame, [arr], True, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, tuple(p), DRAG_RADIUS, (0, 255, 100), 2)
            cv2.circle(frame, tuple(p), 4, (0, 255, 100), -1)


def draw_scoreboard(frame, penalty_counts, tid_to_role, role_to_tid, role_distance_cm):
    """Top-right panel showing role assignments, penalty counts and distance."""
    fw = frame.shape[1]
    panel_w, panel_h = 240, 110
    x0 = fw - panel_w - 10
    y0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "SCOREBOARD", (x0 + 10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 50), 1, cv2.LINE_AA)

    row = y0 + 44
    for role_key, (name, color) in ROLES.items():
        tid   = role_to_tid.get(role_key, "?")
        pen   = penalty_counts.get(role_key, 0)
        dist  = role_distance_cm.get(role_key)
        label = f"{name}  (ID {tid})"
        if role_key in PENALTY_ROLES:
            label += f"  PEN:{pen}"
        if dist is not None:
            label += f"  {dist:.0f}cm"
        cv2.putText(frame, label, (x0 + 10, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        row += 22


def draw_assignment_prompt(frame, pending_role):
    if pending_role is None:
        return
    name, color = ROLES[pending_role]
    msg = f"  Click on {name} in the video  (ESC to cancel)"
    h   = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 58), (frame.shape[1], h - 30), (40, 0, 80), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, msg, (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_penalty_flash(frame, role_key, track_id, penalty_counts):
    name, color = ROLES[role_key]
    cnt = penalty_counts.get(role_key, 0)
    msg = f"  PENALTY – {name} (ID {track_id}) OUT OF BOUNDS!  [#{cnt}]"
    cv2.putText(frame, msg, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3, cv2.LINE_AA)


def draw_help(frame):
    lines = [
        " KEYBOARD SHORTCUTS",
        "  1  : assign next click → Fighter 1",
        "  2  : assign next click → Fighter 2",
        "  R  : assign next click → Referee",
        "  ESC: cancel assignment mode",
        "  X  : clear all role assignments",
        "  D  : start drawing mat boundary",
        "  F  : finish / close boundary",
        "  C  : clear boundary",
        "  S  : save boundary",
        "  P  : toggle pose / bbox tracking",
        "  H  : toggle this help",
        "",
        " MOUSE",
        "  Assignment mode: click a person box",
        "  Normal mode: drag green corner handles",
    ]
    overlay = frame.copy()
    x0, y0, pad, line_h = 20, 20, 10, 22
    box_h = len(lines) * line_h + pad * 2
    cv2.rectangle(overlay, (x0, y0), (x0 + 420, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0 + pad, y0 + pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


def draw_status_bar(frame, drawing_mode, use_pose, pending_role):
    h = frame.shape[0]
    if pending_role:
        mode_str = f"ASSIGN MODE – click {ROLES[pending_role][0]}"
    elif drawing_mode:
        mode_str = "DRAW MODE – click to add corners (F=finish)"
    else:
        mode_str = "EDIT MODE – drag corners | 1/2/R=assign role"
    pose_str = "POSE (ankles)" if use_pose else "BBOX (bottom-centre)"
    bar = f"  {mode_str}   |   {pose_str}   |   H=help"
    cv2.rectangle(frame, (0, h - 28), (frame.shape[1], h), (30, 30, 30), -1)
    cv2.putText(frame, bar, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  FighterTracker
# ─────────────────────────────────────────────────────────────────────────────
class FighterTracker:
    """Holds all tracker state; front-ends feed it frames and input events."""

    def __init__(self, use_pose=False):
        self.polygon_pts  = load_polygon(SAVED_POINTS_FILE)
        self.drawing_mode = False
        self.dragging_idx = None
        self.use_pose     = use_pose
        self.show_help    = False

        self.role_to_tid = {}       # "F1" → track_id
        self.tid_to_role = {}       # track_id → "F1"
        self.role_histograms = {r: deque(maxlen=REID_HISTORY) for r in ROLES}

        self.outside_counters = {}                   # track_id → int
        self.penalty_counts   = {"F1": 0, "F2": 0}    # persistent per role
        self.role_distance_cm = {}                    # "F1" → float | None

        self.pending_role     = None
        self.frame_detections = []    # [(track_id, x1,y1,x2,y2), ...] this frame
        self.last_frame       = None  # copy of the most recent input frame

        self.model = None
        self._log_buf = deque(maxlen=50)

    # ── logging (also surfaced to a UI's log panel) ─────────────────────────
    def log(self, msg):
        print(msg)
        self._log_buf.append(msg)

    def pop_logs(self):
        msgs = list(self._log_buf)
        self._log_buf.clear()
        return msgs

    # ── model lifecycle ──────────────────────────────────────────────────────
    def load_model(self):
        model_name = "yolov8n-pose.pt" if self.use_pose else "yolov8n.pt"
        self.model = YOLO(model_name)
        self.log(f"[INFO] Model: {model_name}")

    def shutdown(self):
        save_polygon(SAVED_POINTS_FILE, self.polygon_pts)

    # ── role assignment / re-ID ──────────────────────────────────────────────
    def assign_role(self, role_key, track_id, frame, x1, y1, x2, y2):
        """Pin role_key to track_id and seed its histogram."""
        old_tid  = self.role_to_tid.get(role_key)
        old_role = self.tid_to_role.get(track_id)
        if old_tid is not None:
            self.tid_to_role.pop(old_tid, None)
        if old_role is not None:
            self.role_to_tid.pop(old_role, None)

        self.role_to_tid[role_key] = track_id
        self.tid_to_role[track_id] = role_key

        hist = extract_histogram(frame, x1, y1, x2, y2)
        if hist is not None:
            self.role_histograms[role_key].clear()
            self.role_histograms[role_key].append(hist)

        name = ROLES[role_key][0]
        self.log(f"[INFO] {name} assigned to tracker ID {track_id}")

    def best_role_match(self, hist):
        """Return (role_key, distance) for the closest known appearance, or (None, 1.0)."""
        best_role, best_dist = None, 1.0
        for role, hist_deq in self.role_histograms.items():
            if not hist_deq:
                continue
            d = min(hist_distance(hist, h) for h in hist_deq)
            if d < best_dist:
                best_dist, best_role = d, role
        if best_role and best_dist < REID_THRESHOLD:
            return best_role, best_dist
        return None, 1.0

    def try_reid(self, track_id, frame, x1, y1, x2, y2):
        """If track_id has no role, try to match it to a known appearance."""
        if track_id in self.tid_to_role:
            return

        hist = extract_histogram(frame, x1, y1, x2, y2)
        if hist is None:
            return

        role, dist = self.best_role_match(hist)
        if role is None:
            return

        old_tid = self.role_to_tid.get(role)
        if old_tid is not None and old_tid != track_id:
            active_ids = {d[0] for d in self.frame_detections}
            if old_tid in active_ids:
                return   # original still visible – don't steal

        self.assign_role(role, track_id, frame, x1, y1, x2, y2)
        self.log(f"[INFO] Re-ID: {ROLES[role][0]} recovered (dist={dist:.3f})")

    # ── input handling (front-end agnostic) ──────────────────────────────────
    def on_mouse_event(self, event, x, y):
        """event: 'down' | 'move' | 'up' (mirrors LBUTTONDOWN/MOUSEMOVE/LBUTTONUP)."""
        if self.drawing_mode:
            if event == "down":
                self.polygon_pts.append([x, y])
            return

        if self.pending_role is not None and event == "down":
            for (tid, x1, y1, x2, y2) in self.frame_detections:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if self.last_frame is not None:
                        self.assign_role(self.pending_role, tid, self.last_frame, x1, y1, x2, y2)
                    self.pending_role = None
                    return
            self.log("[INFO] Assignment cancelled (clicked outside any person).")
            self.pending_role = None
            return

        if event == "down":
            self.dragging_idx = nearest_corner(self.polygon_pts, x, y)
        elif event == "move" and self.dragging_idx is not None:
            self.polygon_pts[self.dragging_idx] = [x, y]
        elif event == "up":
            self.dragging_idx = None

    def on_key(self, key):
        """key: single character ('1','2','r','h', ...) or 'esc'. Case-insensitive."""
        key = key.lower()

        if key == "h":
            self.show_help = not self.show_help
        elif key == "esc":
            self.pending_role = None
            self.log("[INFO] Assignment cancelled.")
        elif key == "1":
            self.pending_role = "F1"
            self.log("[INFO] Click on Fighter 1")
        elif key == "2":
            self.pending_role = "F2"
            self.log("[INFO] Click on Fighter 2")
        elif key == "r":
            self.pending_role = "REF"
            self.log("[INFO] Click on Referee")
        elif key == "x":
            self.role_to_tid.clear()
            self.tid_to_role.clear()
            self.outside_counters.clear()
            self.penalty_counts = {"F1": 0, "F2": 0}
            for dq in self.role_histograms.values():
                dq.clear()
            self.pending_role = None
            self.log("[INFO] All role assignments cleared.")
        elif key == "d":
            self.drawing_mode = True
            self.polygon_pts  = []
        elif key == "f":
            self.drawing_mode = False
            self.log(f"[INFO] Boundary closed ({len(self.polygon_pts)} corners).")
        elif key == "c":
            self.polygon_pts  = []
            self.drawing_mode = False
        elif key == "s":
            save_polygon(SAVED_POINTS_FILE, self.polygon_pts)
            self.log(f"[INFO] Polygon saved to {SAVED_POINTS_FILE}")
        elif key == "p":
            self.use_pose = not self.use_pose
            self.load_model()
            self.outside_counters.clear()
            self.log(f"[INFO] Switched to {'pose' if self.use_pose else 'bbox'} tracking")

    # ── per-frame processing ─────────────────────────────────────────────────
    def process_frame(self, frame, camera):
        """Run detection/tracking/logic on `frame` and draw all overlays in place.

        `camera` must expose get_distance_cm(x, y) for the just-grabbed frame
        (see camera_integration.RealSenseCamera).
        """
        self.last_frame = frame.copy()

        results = self.model.track(
            frame,
            classes=[0],          # persons only
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]

        self.frame_detections.clear()
        kps_list = []
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids   = result.boxes.id.cpu().numpy().astype(int)
            kps_list = (result.keypoints.xy.cpu().numpy()
                        if (self.use_pose and result.keypoints is not None)
                        else [None] * len(ids))

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                self.frame_detections.append((int(tid), x1, y1, x2, y2))

        # Re-ID pass
        for (tid, x1, y1, x2, y2) in self.frame_detections:
            self.try_reid(tid, self.last_frame, x1, y1, x2, y2)

        # Learn appearance for confirmed roles
        for (tid, x1, y1, x2, y2) in self.frame_detections:
            role = self.tid_to_role.get(tid)
            if role:
                h = extract_histogram(self.last_frame, x1, y1, x2, y2)
                if h is not None:
                    self.role_histograms[role].append(h)

        draw_polygon(frame, self.polygon_pts, self.drawing_mode)
        has_poly = len(self.polygon_pts) >= 3
        poly_arr = poly_array(self.polygon_pts) if has_poly else None

        active_penalty_roles = set()
        self.role_distance_cm.clear()

        for (tid, x1, y1, x2, y2), kp in zip(self.frame_detections, kps_list):
            role     = self.tid_to_role.get(tid)
            is_known = role is not None

            if is_known:
                name, color = ROLES[role]
            else:
                name, color = f"Person #{tid}", (140, 140, 140)

            # Real-world distance at the tracking point (ankle if pose, else bbox bottom-centre)
            if self.use_pose and kp is not None:
                dist_pt = tuple(map(int, kp[LEFT_ANKLE]))
            else:
                dist_pt = ((x1 + x2) // 2, y2)
            dist_cm = camera.get_distance_cm(*dist_pt)
            if is_known and dist_cm is not None:
                self.role_distance_cm[role] = dist_cm

            if has_poly and is_known:
                outside_now, debug_pts = check_outside(
                    poly_arr, self.use_pose, kp, x1, y1, x2, y2)

                self.outside_counters[tid] = self.outside_counters.get(tid, 0)
                self.outside_counters[tid] = self.outside_counters[tid] + 1 if outside_now else 0

                penalty_active = (self.outside_counters[tid] > OUTSIDE_THRESHOLD
                                  and role in PENALTY_ROLES)

                if penalty_active:
                    active_penalty_roles.add(role)
                    if self.outside_counters[tid] == OUTSIDE_THRESHOLD + 1:
                        self.penalty_counts[role] = self.penalty_counts.get(role, 0) + 1

                status = "PENALTY" if penalty_active else "INSIDE"
                box_c  = (0, 0, 255) if penalty_active else color

                for (px, py), inside in debug_pts:
                    cv2.circle(frame, (px, py), 7,
                               (0, 255, 0) if inside else (0, 0, 255), -1)
            else:
                status = "–"
                box_c  = color

            thickness = 3 if is_known else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_c, thickness)

            header = f"{name}  [ID {tid}]" if is_known else f"ID {tid}"
            if dist_cm is not None:
                header += f"  {dist_cm:.0f}cm"
            cv2.putText(frame, header, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, box_c, 2, cv2.LINE_AA)
            if has_poly and is_known:
                cv2.putText(frame, status, (x1, y2 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_c, 2, cv2.LINE_AA)

            if self.pending_role is not None:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              (255, 255, 0), 2)

        for role in active_penalty_roles:
            tid = self.role_to_tid.get(role, "?")
            draw_penalty_flash(frame, role, tid, self.penalty_counts)

        draw_scoreboard(frame, self.penalty_counts, self.tid_to_role, self.role_to_tid, self.role_distance_cm)
        draw_assignment_prompt(frame, self.pending_role)
        draw_status_bar(frame, self.drawing_mode, self.use_pose, self.pending_role)
        if self.show_help:
            draw_help(frame)

        return frame
