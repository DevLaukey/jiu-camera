"""
SAM Team Tag Tracker — core logic
==================================
Tracks two teams of tagging athletes (A1/A2 vs B1/B2, plus a referee) over
three mat zones — the central fight square and one tag zone per team — and
records team penalties according to the SAM rule set:

  1. DOUBLE OCCUPANCY   two athletes of the same team inside the fight
                        square for more than DOUBLE_OCC_SECONDS → penalty.
  2. BORDER BREACH      an athlete standing on the white border strip
                        (i.e. outside every zone) for more than
                        BREACH_SECONDS → penalty. This also covers breaches
                        made while transitioning between zones.
  3. QUICK RE-TAG       an athlete who tags out (fight square → own tag
                        zone) and tags back in within RETAG_SECONDS →
                        penalty (prohibits incessant tagging in/out).
  4. GAME OVER          when a team reaches `max_penalties` (default 6,
                        adjustable at runtime with [ and ]) the screen
                        turns red and no further penalties are recorded.

  (The "lower body part used" rule is intentionally NOT implemented: it
  requires action recognition beyond this tracking pipeline and is slated
  to be abolished.)

The class has no dependency on how frames are displayed or how input
arrives. Two front-ends drive it:

  model.py   — local OpenCV window (cv2.imshow + mouse callback)
  server.py  — browser-based UI (MJPEG stream + WebSocket control),
               for headless boards (e.g. a Jetson) controlled remotely.

CONTROLS  (same meaning from either front-end)
  D        draw the FIGHT SQUARE (click corners)
  T        draw TEAM A's tag zone
  Y        draw TEAM B's tag zone
  F        finish / close the zone being drawn
  C        clear ALL zones
  S        save zones → saved_points.json
  P        toggle pose-mode (ankle) vs bbox-centre tracking
  1/2      enter "assign" mode: next click assigns athlete A1 / A2
  3/4      enter "assign" mode: next click assigns athlete B1 / B2
  R        enter "assign" mode: next click assigns the Referee
  ESC      cancel assignment mode
  X        clear ALL role assignments (keeps penalties)
  G        reset the game (penalties, game-over state, rule timers)
  [ / ]    decrease / increase the penalty limit (game-over threshold)
  H        toggle help overlay

MOUSE (normal mode)
  Left-click a person box  → assign current pending role
  Left-click+drag corner   → reposition a zone corner
"""

import json
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
BREACH_SECONDS        = 0.5    # continuously outside every zone → penalty
DOUBLE_OCC_SECONDS    = 3.0    # 2 same-team athletes in fight square → penalty
RETAG_SECONDS         = 5.0    # min. time between tag-out and tag-back-in
ZONE_DWELL_SECONDS    = 0.3    # a zone change must persist this long to count
FLASH_SECONDS         = 2.5    # how long a penalty banner stays on screen
DEFAULT_MAX_PENALTIES = 6      # team penalties that end the game ([ / ] adjust)

DRAG_RADIUS        = 14        # px to grab a polygon corner
REID_HISTORY       = 8         # appearance histograms kept per role
REID_THRESHOLD     = 0.45      # Bhattacharyya distance to accept re-id
SAVED_POINTS_FILE  = "saved_points.json"
WINDOW_NAME        = "SAM Tag Tracker  [H=help]"

LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

# Team definitions  →  (display_name, BGR_color)
TEAMS = {
    "A": ("Team A", (255, 140, 0)),    # blue
    "B": ("Team B", (0,   80, 255)),   # red-orange
}

# Role definitions  →  (display_name, BGR_color)
ROLES = {
    "A1":  ("A1",      (255, 140, 0)),
    "A2":  ("A2",      (255, 210, 90)),
    "B1":  ("B1",      (0,   80, 255)),
    "B2":  ("B2",      (0,  170, 255)),
    "REF": ("Referee", (180, 180, 180)),
}
ROLE_TEAM = {"A1": "A", "A2": "A", "B1": "B", "B2": "B"}   # REF has no team

# Zone definitions  →  (display_name, BGR_color).  Athletes standing inside
# none of them are on the white border strip (or off the mat) = breach.
ZONES = {
    "fight": ("FIGHT",  (0, 255, 0)),
    "tagA":  ("TAG A",  (255, 140, 0)),
    "tagB":  ("TAG B",  (0,  80, 255)),
}
TEAM_TAG_ZONE = {"A": "tagA", "B": "tagB"}

# Which key starts drawing which zone
ZONE_DRAW_KEYS = {"d": "fight", "t": "tagA", "y": "tagB"}


# ─────────────────────────────────────────────
#  Stateless helpers
# ─────────────────────────────────────────────
def load_zones(path: str):
    """Load {zone_key: [[x,y],...]} — accepts the legacy single-list format."""
    empty = {z: [] for z in ZONES}
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return empty
    if isinstance(data, list):            # legacy: one polygon = fight square
        empty["fight"] = data
        return empty
    return {z: data.get(z, []) for z in ZONES}


def save_zones(path: str, zones: dict):
    with open(path, "w") as f:
        json.dump(zones, f)


def poly_array(pts):
    """Return pts as np.int32 array shaped for OpenCV."""
    return np.array(pts, dtype=np.int32)


def nearest_corner(zones, x, y):
    """Return (zone_key, corner_idx) of the nearest grabbable corner, else None."""
    for zkey, pts in zones.items():
        for i, (px, py) in enumerate(pts):
            if abs(px - x) <= DRAG_RADIUS and abs(py - y) <= DRAG_RADIUS:
                return zkey, i
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


def tracking_points(use_pose, kp, x1, y1, x2, y2):
    """Ground-contact points used for zone membership (ankles or bbox base)."""
    if use_pose and kp is not None:
        return [tuple(map(int, kp[LEFT_ANKLE])), tuple(map(int, kp[RIGHT_ANKLE]))]
    return [((x1 + x2) // 2, y2)]


def locate_zone(zone_arrays, pts):
    """Return (zone_key | None, debug_points).

    zone_key: the zone the athlete counts as being in — the fight square wins
    if any point is inside it, then tag zones; None = on the white border /
    off the mat (all points outside every zone).
    debug_points: [((x,y), inside_any_zone_bool)] for drawing.
    """
    per_pt = []
    for pt in pts:
        hit = None
        for zkey in ("fight", "tagA", "tagB"):
            arr = zone_arrays.get(zkey)
            if arr is not None and point_inside(arr, pt):
                hit = zkey
                break
        per_pt.append(hit)

    debug = [(pt, hit is not None) for pt, hit in zip(pts, per_pt)]
    for zkey in ("fight", "tagA", "tagB"):
        if zkey in per_pt:
            return zkey, debug
    return None, debug


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_zones(frame, zones, drawing_zone):
    for zkey, pts in zones.items():
        name, color = ZONES[zkey]
        is_drawing = (zkey == drawing_zone)
        if len(pts) < 2:
            for p in pts:
                cv2.circle(frame, tuple(p), 5, color, -1)
            continue
        arr = poly_array(pts)
        if is_drawing:
            cv2.polylines(frame, [arr], False, (0, 200, 255), 2)
            for p in pts:
                cv2.circle(frame, tuple(p), 5, (0, 200, 255), -1)
        else:
            cv2.polylines(frame, [arr], True, color, 2)
            for p in pts:
                cv2.circle(frame, tuple(p), DRAG_RADIUS, color, 2)
                cv2.circle(frame, tuple(p), 4, color, -1)
        if len(pts) >= 3:
            cx, cy = arr.mean(axis=0).astype(int)
            cv2.putText(frame, name, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_scoreboard(frame, team_penalties, max_penalties,
                    role_to_tid, role_distance_cm):
    """Top-right panel: team penalty tallies plus athlete IDs/distances."""
    fw = frame.shape[1]
    panel_w, panel_h = 260, 190
    x0 = fw - panel_w - 10
    y0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"SCOREBOARD   (limit {max_penalties})", (x0 + 10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 50), 1, cv2.LINE_AA)

    row = y0 + 44
    for team_key, (team_name, color) in TEAMS.items():
        pen = team_penalties.get(team_key, 0)
        cv2.putText(frame, f"{team_name}   PEN {pen}/{max_penalties}",
                    (x0 + 10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        row += 20
        for role_key in [r for r, t in ROLE_TEAM.items() if t == team_key]:
            tid  = role_to_tid.get(role_key, "?")
            dist = role_distance_cm.get(role_key)
            label = f"  {role_key}  ID {tid}"
            if dist is not None:
                label += f"  {dist:.0f}cm"
            cv2.putText(frame, label, (x0 + 10, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ROLES[role_key][1], 1, cv2.LINE_AA)
            row += 18
        row += 6
    ref_tid = role_to_tid.get("REF", "?")
    cv2.putText(frame, f"Referee  ID {ref_tid}", (x0 + 10, row),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ROLES["REF"][1], 1, cv2.LINE_AA)


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


def draw_flashes(frame, flashes):
    """Penalty banners at top-left, newest last."""
    y = 60
    for msg, _expiry in flashes:
        cv2.putText(frame, f"  {msg}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)
        y += 34


def draw_game_over(frame, team_key, team_penalties):
    """Full red overlay signalling the end of the game."""
    overlay = frame.copy()
    overlay[:] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    h, w = frame.shape[:2]
    team_name = TEAMS[team_key][0] if team_key in TEAMS else "?"
    pen = team_penalties.get(team_key, 0)
    lines = ["GAME OVER",
             f"{team_name} reached {pen} penalties",
             "press G to reset"]
    y = h // 2 - 40
    for i, txt in enumerate(lines):
        scale = 1.6 if i == 0 else 0.9
        size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        cv2.putText(frame, txt, ((w - size[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
        y += size[1] + 34


def draw_help(frame):
    lines = [
        " KEYBOARD SHORTCUTS",
        "  1/2  : assign next click -> A1 / A2",
        "  3/4  : assign next click -> B1 / B2",
        "  R    : assign next click -> Referee",
        "  ESC  : cancel assignment mode",
        "  X    : clear all role assignments",
        "  D    : draw FIGHT square",
        "  T/Y  : draw Team A / Team B tag zone",
        "  F    : finish zone   C : clear all zones",
        "  S    : save zones",
        "  [ ]  : penalty limit - / +",
        "  G    : reset game (penalties)",
        "  P    : toggle pose / bbox tracking",
        "  H    : toggle this help",
        "",
        " RULES (automatic)",
        f"  2 same-team in fight > {DOUBLE_OCC_SECONDS:.0f}s   -> penalty",
        f"  on white border > {BREACH_SECONDS:.1f}s          -> penalty",
        f"  re-tag within {RETAG_SECONDS:.0f}s of tag-out    -> penalty",
        "  penalty limit reached           -> game over",
    ]
    overlay = frame.copy()
    x0, y0, pad, line_h = 20, 20, 10, 22
    box_h = len(lines) * line_h + pad * 2
    cv2.rectangle(overlay, (x0, y0), (x0 + 440, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0 + pad, y0 + pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


def draw_status_bar(frame, drawing_zone, use_pose, pending_role):
    h = frame.shape[0]
    if pending_role:
        mode_str = f"ASSIGN MODE – click {ROLES[pending_role][0]}"
    elif drawing_zone:
        mode_str = f"DRAW {ZONES[drawing_zone][0]} – click corners (F=finish)"
    else:
        mode_str = "EDIT MODE – drag corners | 1-4/R=assign role"
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
        self.zones        = load_zones(SAVED_POINTS_FILE)
        self.drawing_zone = None          # zone key being drawn, or None
        self.dragging     = None          # (zone_key, corner_idx) while dragging
        self.use_pose     = use_pose
        self.show_help    = False

        self.role_to_tid = {}       # "A1" → track_id
        self.tid_to_role = {}       # track_id → "A1"
        self.role_histograms = {r: deque(maxlen=REID_HISTORY) for r in ROLES}

        # ── penalty / rules state ────────────────────────────────────────────
        self.team_penalties  = {t: 0 for t in TEAMS}
        self.max_penalties   = DEFAULT_MAX_PENALTIES
        self.game_over       = False
        self.game_over_team  = None
        self.penalty_events  = deque(maxlen=100)   # {t, team, role, rule, detail}
        self.flashes         = deque()             # (msg, monotonic expiry)

        self.breach_since    = {}    # role → monotonic time it went off-zone
        self.breach_latched  = set() # roles already penalised for current breach
        self.double_since    = {t: None for t in TEAMS}
        self.double_latched  = {t: False for t in TEAMS}

        self.zone_stable     = {}    # role → debounced current zone (or None)
        self.zone_pending    = {}    # role → (candidate_zone, since)
        self.last_real_zone  = {}    # role → last non-None stable zone
        self.last_tag_out    = {}    # role → monotonic time of last tag-out

        self.role_distance_cm = {}   # "A1" → float | None

        self.pending_role     = None
        self.frame_detections = []   # [(track_id, x1,y1,x2,y2), ...] this frame
        self.last_frame       = None # copy of the most recent input frame

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
        save_zones(SAVED_POINTS_FILE, self.zones)

    # ── penalties / game state ───────────────────────────────────────────────
    def add_penalty(self, team, role, rule, detail):
        """Record one penalty against `team` unless the game is already over."""
        if self.game_over or team not in self.team_penalties:
            return
        self.team_penalties[team] += 1
        cnt = self.team_penalties[team]
        msg = f"PENALTY {cnt}/{self.max_penalties} {TEAMS[team][0]} - {detail}"
        self.log(f"[PENALTY] {msg}")
        self.penalty_events.append({
            "t": time.time(), "team": team, "role": role,
            "rule": rule, "detail": detail, "count": cnt,
        })
        self.flashes.append((msg, time.monotonic() + FLASH_SECONDS))
        if cnt >= self.max_penalties:
            self.game_over      = True
            self.game_over_team = team
            self.log(f"[GAME] {TEAMS[team][0]} reached {cnt} penalties – GAME OVER")

    def reset_game(self):
        self.team_penalties = {t: 0 for t in TEAMS}
        self.game_over      = False
        self.game_over_team = None
        self.penalty_events.clear()
        self.flashes.clear()
        self._reset_rule_timers()
        self.log("[GAME] Game reset – penalties cleared.")

    def _reset_rule_timers(self):
        self.breach_since.clear()
        self.breach_latched.clear()
        self.double_since   = {t: None for t in TEAMS}
        self.double_latched = {t: False for t in TEAMS}
        self.zone_stable.clear()
        self.zone_pending.clear()
        self.last_real_zone.clear()
        self.last_tag_out.clear()

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
        if self.drawing_zone is not None:
            if event == "down":
                self.zones[self.drawing_zone].append([x, y])
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
            self.dragging = nearest_corner(self.zones, x, y)
        elif event == "move" and self.dragging is not None:
            zkey, idx = self.dragging
            self.zones[zkey][idx] = [x, y]
        elif event == "up":
            self.dragging = None

    def on_key(self, key):
        """key: single character ('1','2','r','h', ...) or 'esc'. Case-insensitive."""
        key = key.lower()

        if key == "h":
            self.show_help = not self.show_help
        elif key == "esc":
            self.pending_role = None
            self.log("[INFO] Assignment cancelled.")
        elif key in ("1", "2", "3", "4", "r"):
            self.pending_role = {"1": "A1", "2": "A2",
                                 "3": "B1", "4": "B2", "r": "REF"}[key]
            self.log(f"[INFO] Click on {ROLES[self.pending_role][0]}")
        elif key == "x":
            self.role_to_tid.clear()
            self.tid_to_role.clear()
            for dq in self.role_histograms.values():
                dq.clear()
            self._reset_rule_timers()
            self.pending_role = None
            self.log("[INFO] All role assignments cleared.")
        elif key in ZONE_DRAW_KEYS:
            self.drawing_zone = ZONE_DRAW_KEYS[key]
            self.zones[self.drawing_zone] = []
            self.log(f"[INFO] Drawing {ZONES[self.drawing_zone][0]} – click corners, F to finish.")
        elif key == "f":
            if self.drawing_zone is not None:
                zname = ZONES[self.drawing_zone][0]
                self.log(f"[INFO] {zname} closed ({len(self.zones[self.drawing_zone])} corners).")
            self.drawing_zone = None
        elif key == "c":
            self.zones = {z: [] for z in ZONES}
            self.drawing_zone = None
            self.log("[INFO] All zones cleared.")
        elif key == "s":
            save_zones(SAVED_POINTS_FILE, self.zones)
            self.log(f"[INFO] Zones saved to {SAVED_POINTS_FILE}")
        elif key == "g":
            self.reset_game()
        elif key == "[":
            self.max_penalties = max(1, self.max_penalties - 1)
            self.log(f"[INFO] Penalty limit set to {self.max_penalties}")
        elif key == "]":
            self.max_penalties = min(30, self.max_penalties + 1)
            self.log(f"[INFO] Penalty limit set to {self.max_penalties}")
        elif key == "p":
            self.use_pose = not self.use_pose
            self.load_model()
            self._reset_rule_timers()
            self.log(f"[INFO] Switched to {'pose' if self.use_pose else 'bbox'} tracking")

    # ── rules engine (per-frame, time-based) ─────────────────────────────────
    def _stable_zone_update(self, role, zone_now, now):
        """Debounce zone membership; return the committed transition
        (old_zone, new_zone) when one happens, else None."""
        stable = self.zone_stable.get(role, zone_now)
        if role not in self.zone_stable:
            self.zone_stable[role] = zone_now
            if zone_now is not None:
                self.last_real_zone[role] = zone_now
            return None

        if zone_now == stable:
            self.zone_pending.pop(role, None)
            return None

        cand, since = self.zone_pending.get(role, (None, now))
        if cand != zone_now:
            self.zone_pending[role] = (zone_now, now)
            return None
        if now - since < ZONE_DWELL_SECONDS:
            return None

        # Commit the transition
        self.zone_pending.pop(role, None)
        self.zone_stable[role] = zone_now
        return stable, zone_now

    def _apply_rules(self, role, team, zone_now, now):
        """Breach + tag rules for one athlete this frame."""
        # 1. Border breach: continuously outside every zone > BREACH_SECONDS.
        if zone_now is None:
            self.breach_since.setdefault(role, now)
            if (now - self.breach_since[role] > BREACH_SECONDS
                    and role not in self.breach_latched):
                self.breach_latched.add(role)
                self.add_penalty(team, role, "border_breach",
                                 f"{role} breached the white border")
        else:
            self.breach_since.pop(role, None)
            self.breach_latched.discard(role)

        # 2. Tag events (on debounced zone transitions).
        transition = self._stable_zone_update(role, zone_now, now)
        if transition is None:
            return
        _old, new = transition
        prev_real = self.last_real_zone.get(role)
        own_tag   = TEAM_TAG_ZONE[team]

        if new == own_tag and prev_real == "fight":
            self.last_tag_out[role] = now
            self.log(f"[INFO] {role} tagged out.")
        elif new == "fight" and prev_real == own_tag:
            out_t = self.last_tag_out.get(role)
            if out_t is not None and now - out_t < RETAG_SECONDS:
                self.add_penalty(team, role, "quick_retag",
                                 f"{role} re-tagged within {now - out_t:.1f}s")
            else:
                self.log(f"[INFO] {role} tagged in.")

        if new is not None:
            self.last_real_zone[role] = new

    def _check_double_occupancy(self, fight_count, now):
        """≥2 same-team athletes in the fight square > DOUBLE_OCC_SECONDS."""
        for team in TEAMS:
            if fight_count.get(team, 0) >= 2:
                if self.double_since[team] is None:
                    self.double_since[team] = now
                elif (now - self.double_since[team] > DOUBLE_OCC_SECONDS
                        and not self.double_latched[team]):
                    self.double_latched[team] = True
                    self.add_penalty(team, None, "double_occupancy",
                                     "2 athletes in the fight square")
            else:
                self.double_since[team]   = None
                self.double_latched[team] = False

    # ── per-frame processing ─────────────────────────────────────────────────
    def process_frame(self, frame, camera):
        """Run detection/tracking/rules on `frame` and draw all overlays in place.

        `camera` must expose get_distance_cm(x, y) for the just-grabbed frame
        (see camera_integration.RealSenseCamera).
        """
        now = time.monotonic()
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

        draw_zones(frame, self.zones, self.drawing_zone)
        zone_arrays = {z: poly_array(pts)
                       for z, pts in self.zones.items() if len(pts) >= 3}
        zones_ready = "fight" in zone_arrays and self.drawing_zone is None

        self.role_distance_cm.clear()
        fight_count = {t: 0 for t in TEAMS}   # team → athletes in fight square

        for (tid, x1, y1, x2, y2), kp in zip(self.frame_detections, kps_list):
            role     = self.tid_to_role.get(tid)
            is_known = role is not None

            if is_known:
                name, color = ROLES[role]
            else:
                name, color = f"Person #{tid}", (140, 140, 140)

            pts = tracking_points(self.use_pose, kp, x1, y1, x2, y2)

            # Real-world distance at the primary tracking point
            dist_cm = camera.get_distance_cm(*pts[0])
            if is_known and dist_cm is not None:
                self.role_distance_cm[role] = dist_cm

            team = ROLE_TEAM.get(role) if is_known else None
            status, box_c = "–", color

            if zones_ready and team is not None:
                zone_now, debug_pts = locate_zone(zone_arrays, pts)

                if zone_now == "fight":
                    fight_count[team] += 1
                self._apply_rules(role, team, zone_now, now)

                breaching = role in self.breach_latched
                status = "OUT!" if zone_now is None else ZONES[zone_now][0]
                box_c  = (0, 0, 255) if breaching else color

                for (px, py), inside in debug_pts:
                    cv2.circle(frame, (px, py), 7,
                               (0, 255, 0) if inside else (0, 0, 255), -1)

            thickness = 3 if is_known else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_c, thickness)

            header = f"{name}  [ID {tid}]" if is_known else f"ID {tid}"
            if dist_cm is not None:
                header += f"  {dist_cm:.0f}cm"
            cv2.putText(frame, header, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, box_c, 2, cv2.LINE_AA)
            if zones_ready and team is not None:
                cv2.putText(frame, status, (x1, y2 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_c, 2, cv2.LINE_AA)

            if self.pending_role is not None:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              (255, 255, 0), 2)

        if zones_ready:
            self._check_double_occupancy(fight_count, now)

        # Expire and draw penalty banners
        while self.flashes and self.flashes[0][1] < now:
            self.flashes.popleft()
        draw_flashes(frame, self.flashes)

        draw_scoreboard(frame, self.team_penalties, self.max_penalties,
                        self.role_to_tid, self.role_distance_cm)
        draw_assignment_prompt(frame, self.pending_role)
        draw_status_bar(frame, self.drawing_zone, self.use_pose, self.pending_role)
        if self.show_help:
            draw_help(frame)
        if self.game_over:
            draw_game_over(frame, self.game_over_team, self.team_penalties)

        return frame
