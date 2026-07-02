"""
Interactive Fighter Boundary Tracker
=====================================
Click a detected person on the video to assign them as Fighter 1,
Fighter 2, or Referee.  Roles survive tracker-ID swaps via an
appearance-based re-ID embedding (histogram).
 
KEYBOARD
  D        start drawing mat boundary (click corners)
  F        finish / close boundary
  C        clear boundary
  S        save boundary → saved_points.json
  P        toggle pose-mode (ankle) vs bbox-centre tracking
  1/2/R    enter "assign" mode: next click assigns Fighter1/2/Referee
  ESC      cancel assignment mode
  X        clear ALL role assignments
  H        toggle help
  Q        quit
 
MOUSE (normal mode)
  Left-click a person box  → assign current pending role
  Left-click+drag corner   → reposition mat corner
"""

import cv2
import numpy as np
import json
import argparse
from collections import deque
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
#  Global UI state
# ─────────────────────────────────────────────
polygon_pts      = []      # list of [x, y]  (working boundary)
drawing_mode     = False   # True while user is clicking new corners
dragging_idx     = None    # index of corner being dragged
use_pose         = False   # True → yolov8n-pose  /  False → yolov8n
show_help        = False

# Role ↔ tracker-ID mapping  (may change when tracker resets)
role_to_tid      = {}       # "F1" → track_id  (int)
tid_to_role      = {}       # track_id → "F1"

# Appearance re-ID: role → deque of histograms
role_histograms  = {r: deque(maxlen=REID_HISTORY) for r in ROLES}

# Penalty state
outside_counters = {}       # track_id → int
penalty_counts   = {"F1": 0, "F2": 0}   # persistent penalty counter per role

# Assignment mode
pending_role     = None     # which role we're about to assign

# Per-frame detection snapshot (filled each frame, read by mouse cb)
frame_detections = []       # list of (track_id, x1,y1,x2,y2)



# ─────────────────────────────────────────────
#  Functions
# ─────────────────────────────────────────────
#  1. Load Polygon
def load_polygon(path: str):
    try:
        with open(path) as f:
            pts = json.load(f)
        print(f"[INFO] Loaded polygon from {path}")
        return pts
    except FileNotFoundError:
        return []

# 2. Save Polygon
def save_polygon(path: str, pts: list):
    with open(path, "w") as f:
        json.dump(pts, f)
    print(f"[INFO] Polygon saved to {path}")

# 3. Create array for polygon
def poly_array(pts):
    """Return pts as np.int32 array shaped for OpenCV."""
    return np.array(pts, dtype=np.int32)
 
def nearest_corner(pts, x, y):
    """Return index of nearest corner within DRAG_RADIUS, else None."""
    for i, (px, py) in enumerate(pts):
        if abs(px - x) <= DRAG_RADIUS and abs(py - y) <= DRAG_RADIUS:
            return i
    return None
   


# ─────────────────────────────────────────────────────────────────────────────
#  Appearance (re-ID)
# ─────────────────────────────────────────────────────────────────────────────
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

def best_role_match(hist):
    """Return (role_key, distance) for the closest known appearance, or None."""
    best_role, best_dist = None, 1.0
    for role, hist_deq in role_histograms.items():
        if not hist_deq:
            continue
        dists = [hist_distance(hist, h) for h in hist_deq]
        d = min(dists)
        if d < best_dist:
            best_dist, best_role = d, role
    if best_role and best_dist < REID_THRESHOLD:
        return best_role, best_dist
    return None, 1.0
 


# ─────────────────────────────────────────────────────────────────────────────
#  Role assignment helpers
# ─────────────────────────────────────────────────────────────────────────────
def assign_role(role_key, track_id, frame, x1, y1, x2, y2):
    """Pin role_key to track_id and seed its histogram."""
    global role_to_tid, tid_to_role
 
    # Remove stale mappings for this role or this track_id
    old_tid  = role_to_tid.get(role_key)
    old_role = tid_to_role.get(track_id)
    if old_tid is not None:
        tid_to_role.pop(old_tid, None)
    if old_role is not None:
        role_to_tid.pop(old_role, None)
 
    role_to_tid[role_key]  = track_id
    tid_to_role[track_id]  = role_key
 
    hist = extract_histogram(frame, x1, y1, x2, y2)
    if hist is not None:
        role_histograms[role_key].clear()
        role_histograms[role_key].append(hist)
 
    name = ROLES[role_key][0]
    print(f"[INFO] {name} assigned to tracker ID {track_id}")
 
def try_reid(track_id, frame, x1, y1, x2, y2):
    """If track_id has no role, try to match it to a known appearance."""
    if track_id in tid_to_role:
        return   # already assigned
 
    hist = extract_histogram(frame, x1, y1, x2, y2)
    if hist is None:
        return
 
    role, dist = best_role_match(hist)
    if role is None:
        return
 
    # Make sure that role isn't already claimed by a current tracker
    old_tid = role_to_tid.get(role)
    if old_tid is not None and old_tid != track_id:
        # Only steal if the old tid is absent this frame
        active_ids = {d[0] for d in frame_detections}
        if old_tid in active_ids:
            return   # original still visible – don't steal
 
    assign_role(role, track_id, frame, x1, y1, x2, y2)
    print(f"[INFO] Re-ID: {ROLES[role][0]} recovered (dist={dist:.3f})")
 
 

# ─────────────────────────────────────────────────────────────────────────────
#  Mouse callback
# ─────────────────────────────────────────────────────────────────────────────
last_frame_ref = [None]   # mutable container so callback can read latest frame

def on_mouse(event, x, y, flags, param):
    global polygon_pts, drawing_mode, dragging_idx, pending_role
 
    # ── Drawing mode ──────────────────────────────────────────────────────────
    if drawing_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_pts.append([x, y])
        return
 
    # ── Assignment mode: click picks the person ───────────────────────────────
    if pending_role is not None and event == cv2.EVENT_LBUTTONDOWN:
        # Find which bounding box was clicked
        for (tid, x1, y1, x2, y2) in frame_detections:
            if x1 <= x <= x2 and y1 <= y <= y2:
                frm = last_frame_ref[0]
                if frm is not None:
                    assign_role(pending_role, tid, frm, x1, y1, x2, y2)
                pending_role = None
                return
        # Clicked outside all boxes – cancel
        pending_role = None
        return
 
    # ── Normal mode: drag corners ─────────────────────────────────────────────
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging_idx = nearest_corner(polygon_pts, x, y)
 
    elif event == cv2.EVENT_MOUSEMOVE and dragging_idx is not None:
        polygon_pts[dragging_idx] = [x, y]
 
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = None
 


# ─────────────────────────────────────────────────────────────────────────────
#  Boundary / penalty helpers
# ─────────────────────────────────────────────────────────────────────────────
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
 
def draw_scoreboard(frame, penalty_counts, tid_to_role, role_to_tid):
    """Top-right panel showing role assignments and penalty counts."""
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
        label = f"{name}  (ID {tid})"
        if role_key in PENALTY_ROLES:
            label += f"  PEN:{pen}"
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
 
def draw_penalty_flash(frame, role_key, track_id):
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
        "  Q  : quit",
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
    bar = f"  {mode_str}   |   {pose_str}   |   H=help  Q=quit"
    cv2.rectangle(frame, (0, h - 28), (frame.shape[1], h), (30, 30, 30), -1)
    cv2.putText(frame, bar, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)




# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global polygon_pts, drawing_mode, use_pose, show_help
    global pending_role, outside_counters, penalty_counts
    global role_to_tid, tid_to_role, frame_detections
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://192.168.1.70:8080/video")
    parser.add_argument("--pose",   action="store_true")
    args = parser.parse_args()
    use_pose = args.pose


    #**********************************
    # Load saved boundary
    #**********************************
    polygon_pts = load_polygon(SAVED_POINTS_FILE)


    #**********************************
    # Load model
    #**********************************
    model_name = "yolov8n-pose.pt" if use_pose else "yolov8n.pt"
    model = YOLO(model_name)
    print(f"[INFO] Model: {model_name}")


    #**********************************
    # Open video
    #**********************************
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source : {source}")
        return

    # Setup Window and Mouse Callback
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print("[INFO] Press H for help, Q to quit.")

 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
 
        last_frame_ref[0] = frame.copy()
 
        # ── YOLO inference ───────────────────────────────────────────────────
        results = model.track(
            frame,
            classes=[0],          # persons only
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]
 
        # ── Build per-frame detection list ───────────────────────────────────
        frame_detections.clear()
        kps_list = []
        if result.boxes.id is not None:
            boxes    = result.boxes.xyxy.cpu().numpy()
            ids      = result.boxes.id.cpu().numpy().astype(int)
            kps_list = (result.keypoints.xy.cpu().numpy()
                        if (use_pose and result.keypoints is not None)
                        else [None] * len(ids))
 
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                frame_detections.append((int(tid), x1, y1, x2, y2))
 
        # ── Re-ID pass ───────────────────────────────────────────────────────
        for (tid, x1, y1, x2, y2) in frame_detections:
            try_reid(tid, last_frame_ref[0], x1, y1, x2, y2)
 
        # ── Update histograms for confirmed roles (learning) ─────────────────
        for (tid, x1, y1, x2, y2) in frame_detections:
            role = tid_to_role.get(tid)
            if role:
                h = extract_histogram(last_frame_ref[0], x1, y1, x2, y2)
                if h is not None:
                    role_histograms[role].append(h)
 
        # ── Draw mat ─────────────────────────────────────────────────────────
        draw_polygon(frame, polygon_pts, drawing_mode)
 
        has_poly = len(polygon_pts) >= 3
        poly_arr = poly_array(polygon_pts) if has_poly else None

 
        # ── Draw detections ──────────────────────────────────────────────────
        active_penalty_roles = set()
 
        for (tid, x1, y1, x2, y2), kp in zip(frame_detections, kps_list):
            role     = tid_to_role.get(tid)
            is_known = role is not None
 
            if is_known:
                name, color = ROLES[role]
            else:
                name, color = f"Person #{tid}", (140, 140, 140)
 
            # Boundary test
            if has_poly and is_known:
                outside_now, debug_pts = check_outside(
                    poly_arr, use_pose, kp, x1, y1, x2, y2)
 
                if tid not in outside_counters:
                    outside_counters[tid] = 0
                outside_counters[tid] = outside_counters[tid] + 1 if outside_now else 0
 
                penalty_active = (outside_counters[tid] > OUTSIDE_THRESHOLD
                                  and role in PENALTY_ROLES)
 
                if penalty_active:
                    active_penalty_roles.add(role)
                    # Increment persistent penalty count only on the frame it crosses threshold
                    if outside_counters[tid] == OUTSIDE_THRESHOLD + 1:
                        penalty_counts[role] = penalty_counts.get(role, 0) + 1
 
                status = "PENALTY" if penalty_active else "INSIDE"
                box_c  = (0, 0, 255) if penalty_active else color
 
                for (px, py), inside in debug_pts:
                    cv2.circle(frame, (px, py), 7,
                               (0, 255, 0) if inside else (0, 0, 255), -1)
            else:
                status = "–"
                box_c  = color
 
            # Bounding box
            thickness = 3 if is_known else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_c, thickness)
 
            # Label: role name + tracker id
            header = f"{name}  [ID {tid}]" if is_known else f"ID {tid}"
            cv2.putText(frame, header, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, box_c, 2, cv2.LINE_AA)
            if has_poly and is_known:
                cv2.putText(frame, status, (x1, y2 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_c, 2, cv2.LINE_AA)
 
            # Assignment hint: highlight when in assignment mode
            if pending_role is not None:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              (255, 255, 0), 2)
 
        # Penalty flash banners
        for role in active_penalty_roles:
            tid = role_to_tid.get(role, "?")
            draw_penalty_flash(frame, role, tid)
 
        # ── UI overlays ──────────────────────────────────────────────────────
        draw_scoreboard(frame, penalty_counts, role_to_tid, tid_to_role)
        draw_assignment_prompt(frame, pending_role)
        draw_status_bar(frame, drawing_mode, use_pose, pending_role)
        if show_help:
            draw_help(frame)
 
        cv2.imshow(WINDOW_NAME, frame)
 
        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == 27:                       # ESC
            pending_role = None
            print("[INFO] Assignment cancelled.")
        elif key == ord('1'):
            pending_role = "F1"
            print("[INFO] Click on Fighter 1")
        elif key == ord('2'):
            pending_role = "F2"
            print("[INFO] Click on Fighter 2")
        elif key in (ord('r'), ord('R')):
            pending_role = "REF"
            print("[INFO] Click on Referee")
        elif key in (ord('x'), ord('X')):
            role_to_tid.clear()
            tid_to_role.clear()
            outside_counters.clear()
            penalty_counts.clear()
            for dq in role_histograms.values():
                dq.clear()
            pending_role = None
            print("[INFO] All role assignments cleared.")
        elif key == ord('d'):
            drawing_mode = True
            polygon_pts  = []
        elif key == ord('f'):
            drawing_mode = False
            print(f"[INFO] Boundary closed ({len(polygon_pts)} corners).")
        elif key == ord('c'):
            polygon_pts  = []
            drawing_mode = False
        elif key == ord('s'):
            save_polygon(SAVED_POINTS_FILE, polygon_pts)
        elif key == ord('p'):
            use_pose   = not use_pose
            model_name = "yolov8n-pose.pt" if use_pose else "yolov8n.pt"
            model      = YOLO(model_name)
            outside_counters.clear()
            print(f"[INFO] Switched to {model_name}")
 
    cap.release()
    cv2.destroyAllWindows()
    save_polygon(SAVED_POINTS_FILE, polygon_pts)
 
 
if __name__ == "__main__":
    main()







