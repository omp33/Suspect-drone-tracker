"""
Ground Camera Threat Detector (Coordinated Dual-Pass Gun-Keypoint Fusion + Pose + ByteTrack)
=============================================================================================
Combines YOLOv8-Pose + ByteTrack with a security-audited CCTV firearm detector
using Full-Frame + Pose-Guided Wrist-ROI dual inference and directional forearm-grip
attribution for sub-second threat confirmation:
- Tier 1 (GUN+POSE):    Weapon in wrist/grip ray + aiming/ready arm (1-frame instant lock)
- Tier 2 (GUN-IN-HAND): High-conf pistol (>=0.60, 1-frame) or >=weapon_conf (2-frame lock)
                        Catches waist/chest/downward/frontal holds without raised arms
- Tier 3 (POSE):        Strict scale-invariant aiming pose fallback (occluded/tiny firearm)

Usage:
    python detector.py --source path/to/video.mp4
    python detector.py --source 0 --no-display --save-video ground_out.mp4
"""

import os
import time
import argparse
from collections import defaultdict, deque
from threading import Thread, Lock

import cv2
import numpy as np
from ultralytics import YOLO

from reid_utils import (
    load_osnet_model,
    ensure_cctv_weapon_model,
    ensure_weapon_onnx_model,
    verify_pytorch_checkpoint_safety,
    extract_reid_feature,
    extract_outfit_color_feature,
    draw_hud_panel,
    draw_label_pill,
    draw_corner_box,
)

cv2.setNumThreads(1)
_save_lock = Lock()

WEAPON_CLASS_NAMES = {"pistol", "gun", "handgun", "rifle", "firearm", "weapon"}


def _point_to_box_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance from point (px, py) to axis-aligned rectangle [x1, y1, x2, y2] (0 if inside)."""
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def get_arm_grips(keypoints: np.ndarray, box: np.ndarray, conf_thresh: float = 0.40) -> list:
    """
    Build per-arm grip rays from wrist + forearm direction so hand/barrel reach
    along the pointing axis is modeled accurately while keeping off-axis false positives out.
    Returns list of dicts: {"side": "L"|"R", "wrist": np.ndarray, "ray_pts": [np.ndarray, ...]}.
    """
    kp = np.asarray(keypoints, dtype=np.float32)
    box_h = max(1.0, float(box[3] - box[1]))
    torso_h = max(box_h * 0.30, 25.0)
    grips = []
    for side, e_idx, w_idx in (("L", 7, 9), ("R", 8, 10)):
        if kp.shape[0] <= w_idx or kp[w_idx, 2] < conf_thresh:
            continue
        wrist = kp[w_idx, :2]
        ray_pts = [wrist]
        if kp[e_idx, 2] >= conf_thresh:
            forearm = wrist - kp[e_idx, :2]
            flen = float(np.linalg.norm(forearm))
            if flen > 1e-4:
                reach = 0.45 * min(flen, 0.55 * torso_h)
                u = forearm / flen
                for frac in (0.35, 0.70, 1.0):
                    ray_pts.append(wrist + (frac * reach) * u)
        grips.append({"side": side, "wrist": wrist, "ray_pts": ray_pts})
    return grips


def build_wrist_roi(kp_np: np.ndarray, box_np: np.ndarray, frame_w: int, frame_h: int):
    """Build a padded crop bounding wrists & forearms for small distant firearm recovery."""
    kp = np.asarray(kp_np, dtype=np.float32)
    pts = [kp[k, :2] for k in (7, 8, 9, 10) if kp.shape[0] > k and kp[k, 2] >= 0.30]
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    ph = max(1.0, float(box_np[3] - box_np[1]))
    margin = max(56.0, 0.30 * ph)
    rx1 = max(0, int(min(xs) - margin))
    rx2 = min(frame_w, int(max(xs) + margin))
    ry1 = max(0, int(min(ys) - margin))
    ry2 = min(frame_h, int(max(ys) + margin))
    if rx2 - rx1 < 24 or ry2 - ry1 < 24:
        return None
    return (rx1, ry1, rx2, ry2)


def deduplicate_weapons(detections: list, iou_thresh: float = 0.45) -> list:
    """Sort weapon detections by confidence and suppress overlapping Pass 1 / Pass 2 duplicates."""
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda x: x[4], reverse=True)
    kept = []
    for cand in ordered:
        cx1, cy1, cx2, cy2 = cand[:4]
        ca = max(1.0, (cx2 - cx1) * (cy2 - cy1))
        dup = False
        for k in kept:
            ix1, iy1 = max(cx1, k[0]), max(cy1, k[1])
            ix2, iy2 = min(cx2, k[2]), min(cy2, k[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                ua = ca + max(1.0, (k[2] - k[0]) * (k[3] - k[1])) - inter
                if inter / max(1.0, ua) >= iou_thresh:
                    dup = True
                    break
        if not dup:
            kept.append(cand)
    return kept


def assign_weapons_to_people(weapon_detections: list, person_entries: list) -> dict:
    """
    1-to-1 forearm-grip ray attribution: assigns each detected weapon box to at most ONE
    person & arm side whose grip segment [wrist -> projected barrel reach] falls within
    tight distance of the box (filtering out oversized torso/background hallucinations).

    Returns: dict[track_id] -> list of (wbox_tuple, arm_side, dist_px)
    """
    assigned = defaultdict(list)
    if not weapon_detections or not person_entries:
        return assigned

    for wbox in weapon_detections:
        wx1, wy1, wx2, wy2, _w_conf, _w_name = wbox
        gw = max(1.0, wx2 - wx1)
        gh = max(1.0, wy2 - wy1)
        g_area = gw * gh
        g_cx, g_cy = 0.5 * (wx1 + wx2), 0.5 * (wy1 + wy2)
        grip_radius = max(16.0, 0.28 * max(gw, gh))

        best = None  # (score, tid, side, dist)
        for tid, pbox, arm_grips in person_entries:
            pw = max(1.0, float(pbox[2] - pbox[0]))
            ph = max(1.0, float(pbox[3] - pbox[1]))
            p_area = pw * ph
            if g_area > 0.14 * p_area or max(gw, gh) > 0.45 * ph:
                continue  # Reject oversized person/jacket/prop false positives
            for arm in arm_grips:
                min_d = min(_point_to_box_dist(float(pt[0]), float(pt[1]), wx1, wy1, wx2, wy2) for pt in arm["ray_pts"])
                if min_d <= grip_radius:
                    # Tie-break by center proximity when multiple points touch the box
                    cd = float(np.hypot(float(arm["ray_pts"][-1][0]) - g_cx, float(arm["ray_pts"][-1][1]) - g_cy))
                    rank = (min_d, cd)
                    if best is None or rank < best[0]:
                        best = (rank, tid, arm["side"], min_d)

        if best is not None:
            assigned[best[1]].append((wbox, best[2], best[3]))
    return assigned


def analyze_pose_threat(keypoints: np.ndarray, box: np.ndarray, conf_thresh: float = 0.55):
    """
    Scale-invariant aiming-pose analysis on 17 COCO keypoints:
    1. Strict side/3-quarter aiming (>= 140 deg extended outward at chest/shoulder height)
    2. Strict 2-hand raised frontal stance (converging wrists above elbows in front of upper chest)
    3. Semi-ready arm metadata (bent/forward/converging hold >= 105 deg) used strictly
       to boost held-weapon detections into instant Tier-1 GUN+POSE confirmation.
    """
    kp = np.asarray(keypoints, dtype=np.float32)
    empty = {
        "is_threat": False, "pose_score": 0.0, "confidence": "NONE",
        "details": [], "semi_ready_arms": set(), "two_hand_close": False,
    }
    if kp.shape[0] < 13:
        return empty

    def pt(idx, c=conf_thresh):
        return kp[idx, :2] if kp[idx, 2] >= c else None

    ls, rs = pt(5), pt(6)
    le, re = pt(7), pt(8)
    lw, rw = pt(9), pt(10)
    lh, rh = pt(11), pt(12)

    box_h = max(1.0, float(box[3] - box[1]))
    torso_candidates = []
    if ls is not None and lh is not None:
        torso_candidates.append(np.linalg.norm(ls - lh))
    if rs is not None and rh is not None:
        torso_candidates.append(np.linalg.norm(rs - rh))
    torso_h = max(float(np.mean(torso_candidates)) if torso_candidates else 0.0, box_h * 0.28, 25.0)
    sw = float(np.linalg.norm(ls - rs)) if (ls is not None and rs is not None) else 0.0

    semi_ready_arms = set()

    def check_arm(side, shoulder, elbow, wrist):
        if shoulder is None or elbow is None or wrist is None:
            return False, None, 0.0
        v1 = shoulder - elbow
        v2 = wrist - elbow
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        if mag < 1e-5:
            return False, None, 0.0
        cos_a = np.clip(np.dot(v1, v2) / mag, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_a)))

        dx_wrist = abs(float(wrist[0] - shoulder[0]))
        dx_elbow = abs(float(elbow[0] - shoulder[0]))
        dy_wrist = float(wrist[1] - shoulder[1])

        is_straight = angle >= 140.0
        is_outward = (dx_wrist > dx_elbow * 1.04) and (dx_wrist > 0.32 * torso_h)
        is_level = -0.35 * torso_h <= dy_wrist <= 0.52 * torso_h

        # Semi-ready arm stance (only active when paired with a weapon in that hand)
        if angle >= 105.0 and (-0.45 * torso_h <= dy_wrist <= 0.90 * torso_h):
            semi_ready_arms.add(side)

        ang_score = max(0.0, min(1.0, (angle - 110.0) / 70.0))
        ext_score = max(0.0, min(1.0, dx_wrist / (0.60 * torso_h)))
        lvl_score = max(0.0, min(1.0, 1.0 - (abs(dy_wrist) / (0.90 * torso_h))))
        arm_score = float(ang_score * ext_score * lvl_score)
        return (is_straight and is_outward and is_level), angle, arm_score

    l_aim, l_ang, l_score = check_arm("L", ls, le, lw)
    r_aim, r_ang, r_score = check_arm("R", rs, re, rw)

    front_2h = False
    front_2h_score = 0.0
    two_hand_close = False
    # Check close two-hand wrist convergence (even at waist/chest low-ready)
    lw_soft, rw_soft = pt(9, 0.40), pt(10, 0.40)
    if ls is not None and rs is not None and lw_soft is not None and rw_soft is not None and sw > 0.14 * box_h:
        w_dist_soft = float(np.linalg.norm(lw_soft - rw_soft))
        if w_dist_soft <= 0.85 * sw:
            two_hand_close = True

    if ls is not None and rs is not None and lw is not None and rw is not None and sw > 0.16 * box_h:
        w_dist = float(np.linalg.norm(lw - rw))
        mid_sh = 0.5 * (ls + rs)
        mid_wr = 0.5 * (lw + rw)
        dy_wr = float(mid_wr[1] - mid_sh[1])
        raised_above_elbows = (
            (le is not None and lw[1] <= le[1] + 0.05 * torso_h)
            and (re is not None and rw[1] <= re[1] + 0.05 * torso_h)
        )
        if (
            w_dist <= 0.50 * sw
            and (-0.20 * torso_h <= dy_wr <= 0.45 * torso_h)
            and abs(mid_wr[0] - mid_sh[0]) <= 0.65 * sw
            and raised_above_elbows
        ):
            front_2h = True
            front_2h_score = max(0.0, min(1.0, 1.0 - (w_dist / (0.65 * sw))))

    pose_score = max(l_score, r_score, front_2h_score)
    details = []
    if l_aim:
        details.append(f"L-Arm Aim ({l_ang:.0f}deg)")
    if r_aim:
        details.append(f"R-Arm Aim ({r_ang:.0f}deg)")
    if front_2h:
        details.append("2-Hand Front Stance")

    if front_2h or (l_aim and r_aim):
        conf_lvl = "HIGH (2-Hand)"
    elif l_aim or r_aim:
        conf_lvl = "MEDIUM (1-Hand)"
    else:
        conf_lvl = "NONE"

    return {
        "is_threat": (l_aim or r_aim or front_2h),
        "pose_score": pose_score,
        "confidence": conf_lvl,
        "details": details,
        "semi_ready_arms": semi_ready_arms,
        "two_hand_close": two_hand_close,
    }


def crop_person(frame: np.ndarray, bbox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty(0, dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def save_criminal_data(output_dir: str, suspect_id: int, reid_feature: np.ndarray, cropped_image: np.ndarray, frame_idx: int):
    img_dir = os.path.join(output_dir, "images")
    feat_dir = os.path.join(output_dir, "features")
    with _save_lock:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(feat_dir, exist_ok=True)
        tag = f"criminal_{suspect_id}_0_frame_{frame_idx}"
        cv2.imwrite(os.path.join(img_dir, f"{tag}.jpg"), cropped_image)
        np.save(os.path.join(feat_dir, f"{tag}.npy"), reid_feature)


def match_existing_suspect(feat: np.ndarray, color_feat: np.ndarray, saved_profiles: dict):
    for sid, prof in saved_profiles.items():
        r_sim = float(np.dot(feat, prof["reid"]))
        c_sim = float(np.dot(color_feat, prof["color"]))
        if r_sim >= 0.72 and c_sim >= 0.65:
            return sid
    return None


def _extract_weapon_boxes(w_res, model_names, offset_x: float = 0.0, offset_y: float = 0.0) -> list:
    """Parse Ultralytics weapon predictions into canonical (x1, y1, x2, y2, conf, cls_name) tuples."""
    dets = []
    for wr in w_res:
        if wr.boxes is None:
            continue
        w_boxes = wr.boxes.xyxy.cpu().numpy() if hasattr(wr.boxes.xyxy, "cpu") else np.asarray(wr.boxes.xyxy)
        w_cls = wr.boxes.cls.int().cpu().tolist()
        w_cf = wr.boxes.conf.cpu().tolist()
        for wb, wc, cf in zip(w_boxes, w_cls, w_cf):
            cname = str(model_names.get(int(wc), wc))
            if cname.lower() in WEAPON_CLASS_NAMES:
                dets.append((
                    float(wb[0]) + offset_x,
                    float(wb[1]) + offset_y,
                    float(wb[2]) + offset_x,
                    float(wb[3]) + offset_y,
                    float(cf),
                    cname,
                ))
    return dets


def _load_weapon_model(weapon_arg: str):
    """Resolve and security-verify requested weapon detector ('auto', 'onnx', 'none', or path)."""
    if not weapon_arg or weapon_arg.lower() == "none":
        return None
    low = weapon_arg.lower()
    if low == "auto":
        w_path = ensure_cctv_weapon_model()
    elif low == "onnx":
        w_path = ensure_weapon_onnx_model()
    elif os.path.isfile(weapon_arg):
        if weapon_arg.endswith(".pt") and not verify_pytorch_checkpoint_safety(weapon_arg):
            raise RuntimeError(f"Blocked unsafe PyTorch checkpoint: {weapon_arg}")
        w_path = weapon_arg
    else:
        print(f"Warning: weapon model '{weapon_arg}' not found; running pose-only mode.")
        return None
    if w_path and os.path.isfile(w_path):
        wm = YOLO(w_path, task="detect")
        print(f"Loaded verified weapon detector: {w_path} (classes: {wm.names})")
        return wm
    return None


def run_detector(args):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    pose_model = YOLO(args.pose_model)
    weapon_model = _load_weapon_model(args.weapon_model)
    reid_model, device = load_osnet_model(args.osnet_weights)

    tracker_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack_custom.yaml")
    if not os.path.isfile(tracker_cfg):
        tracker_cfg = "bytetrack.yaml"

    writer = None
    if args.save_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not args.no_display:
        cv2.namedWindow("Ground Threat Detector", cv2.WINDOW_NORMAL)

    pose_windows = defaultdict(lambda: deque(maxlen=args.persistence))
    gun_windows = defaultdict(lambda: deque(maxlen=args.persistence))
    both_windows = defaultdict(lambda: deque(maxlen=args.persistence))
    confirmed_tiers = {}  # tid -> tier string
    track_to_suspect = {}
    saved_profiles = {}
    threads = []
    cached_people_draw = []
    cached_weapon_draw = []

    internal_wconf = min(0.22, args.weapon_conf)
    frame_count = 0
    max_threats = 0
    yolo_times, weapon_times, reid_times, frame_times = [], [], [], []

    print(f"Starting ground threat detection ({args.pose_model} + ByteTrack) on: {args.source} (device: {device})")

    while cap.isOpened():
        t_frame = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()
        fh, fw = frame.shape[:2]

        if frame_count % args.frame_skip == 0:
            t0 = time.perf_counter()
            results = pose_model.track(
                frame, persist=True, tracker=tracker_cfg, conf=args.conf, verbose=False
            )
            yolo_times.append((time.perf_counter() - t0) * 1000.0)

            # Collect tracked people & forearm grip rays
            tracked_people = []
            person_entries_for_match = []
            for r in results:
                if r.keypoints is None or r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.asarray(r.boxes.xyxy)
                kps = r.keypoints.data.cpu().numpy() if hasattr(r.keypoints.data, "cpu") else np.asarray(r.keypoints.data)
                ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else list(range(len(boxes)))
                for tid, kp_np, box_np in zip(ids, kps, boxes):
                    arm_grips = get_arm_grips(kp_np, box_np, conf_thresh=0.40)
                    tracked_people.append((tid, kp_np, box_np, arm_grips))
                    person_entries_for_match.append((tid, box_np, arm_grips))

            # Dual-Pass Weapon Inference: Pass 1 (Full Frame) + Pass 2 (Wrist-ROI zoom)
            weapon_detections = []
            if weapon_model is not None:
                tw = time.perf_counter()
                w_res = weapon_model(frame, conf=internal_wconf, verbose=False)
                weapon_detections = _extract_weapon_boxes(w_res, weapon_model.names)

                pass1_match = assign_weapons_to_people(weapon_detections, person_entries_for_match)
                if args.wrist_roi:
                    roi_ran = 0
                    for tid, kp_np, box_np, _arm_grips in tracked_people:
                        if roi_ran >= args.max_roi_people:
                            break
                        best_p1 = max((g[0][4] for g in pass1_match.get(tid, [])), default=0.0)
                        if best_p1 >= 0.50:
                            continue  # Already crisp on full frame
                        roi = build_wrist_roi(kp_np, box_np, fw, fh)
                        if roi is None:
                            continue
                        rx1, ry1, rx2, ry2 = roi
                        roi_res = weapon_model(frame[ry1:ry2, rx1:rx2], conf=internal_wconf, verbose=False)
                        weapon_detections.extend(_extract_weapon_boxes(roi_res, weapon_model.names, rx1, ry1))
                        roi_ran += 1

                weapon_detections = deduplicate_weapons(weapon_detections, iou_thresh=0.45)
                weapon_times.append((time.perf_counter() - tw) * 1000.0)

            # 1-to-1 forearm grip ray assignment
            raw_person_weapons = assign_weapons_to_people(weapon_detections, person_entries_for_match)

            # Analyze pose + apply Pose-Prior gate on low-conf weapons & compute coordinated state
            person_states = []
            held_box_map = {}  # wbox -> arm_side
            for tid, kp_np, box_np, arm_grips in tracked_people:
                analysis = analyze_pose_threat(kp_np, box_np, conf_thresh=args.kp_conf)
                candidates = raw_person_weapons.get(tid, [])
                held_guns = []
                for wb, arm_side, dist in candidates:
                    w_cf = wb[4]
                    strict_or_2h = analysis["is_threat"] or (
                        analysis["two_hand_close"] and arm_side in analysis["semi_ready_arms"]
                    )
                    if w_cf >= args.weapon_conf or (w_cf >= 0.28 and strict_or_2h):
                        held_guns.append((wb, arm_side, dist))
                        held_box_map[wb] = arm_side
                person_states.append((tid, kp_np, box_np, arm_grips, analysis, held_guns))

            cached_weapon_draw = []
            for wb in weapon_detections:
                if wb in held_box_map:
                    cached_weapon_draw.append((wb, True, held_box_map[wb]))
                elif wb[4] >= max(0.50, args.weapon_conf):
                    cached_weapon_draw.append((wb, False, ""))

            cached_people_draw = []
            for tid, _kp_np, box_np, arm_grips, analysis, held_guns in person_states:
                has_gun = len(held_guns) > 0
                has_pose = analysis["is_threat"]
                gun_conf = max((g[0][4] for g in held_guns), default=0.0)
                arm_semi_ready = any(arm_s in analysis["semi_ready_arms"] for _, arm_s, _ in held_guns)
                has_ready_with_gun = has_gun and (
                    has_pose
                    or (gun_conf >= 0.45 and (analysis["two_hand_close"] or arm_semi_ready))
                )

                pose_windows[tid].append(1 if has_pose else 0)
                gun_windows[tid].append(1 if has_gun else 0)
                both_windows[tid].append(1 if has_ready_with_gun else 0)

                p_hits = sum(pose_windows[tid])
                g_hits = sum(gun_windows[tid])
                b_hits = sum(both_windows[tid])

                # Complementary coordinated threat probability
                pose_prob = 0.45 * analysis["pose_score"] + 0.55 * min(1.0, p_hits / float(args.confirm_frames))
                gun_prob = (
                    0.50 * gun_conf + 0.50 * min(1.0, g_hits / float(max(1, args.gun_confirm)))
                    if (g_hits > 0 or has_gun)
                    else 0.0
                )
                if b_hits > 0 or has_ready_with_gun:
                    live_prob = 100.0 * min(1.0, max(pose_prob, gun_prob, 0.72) + 0.15)
                else:
                    live_prob = 100.0 * max(pose_prob, gun_prob)

                # Rapid Sub-Second 3-Tier Confirmation + Hysteresis
                # 1-frame instant lock ONLY on same-frame Strict Aiming Pose + Held Gun (or ultra-high conf >= 0.85);
                # otherwise 2-frame lock (~0.07-0.13s) kills single-frame hand-blur/shadow false positives.
                same_frame_dual = has_pose and has_gun
                if same_frame_dual or b_hits >= 2:
                    confirmed_tiers[tid] = "GUN+POSE"
                elif (has_gun and gun_conf >= args.instant_gun_conf) or g_hits >= args.gun_confirm:
                    confirmed_tiers[tid] = "GUN-IN-HAND"
                elif p_hits >= args.confirm_frames:
                    confirmed_tiers[tid] = "POSE"
                elif tid in confirmed_tiers:
                    if g_hits == 0 and b_hits == 0 and p_hits < max(1, args.confirm_frames // 2):
                        del confirmed_tiers[tid]

                is_confirmed = tid in confirmed_tiers
                if is_confirmed:
                    tier = confirmed_tiers[tid]
                    if tid not in track_to_suspect:
                        crop = crop_person(frame, box_np)
                        if crop.size > 0:
                            t_r = time.perf_counter()
                            feat = extract_reid_feature(crop, reid_model, device)
                            color_feat = extract_outfit_color_feature(crop)
                            reid_times.append((time.perf_counter() - t_r) * 1000.0)
                            if feat is not None:
                                existing_sid = match_existing_suspect(feat, color_feat, saved_profiles)
                                sid = existing_sid if existing_sid is not None else tid
                                track_to_suspect[tid] = sid
                                if existing_sid is None:
                                    saved_profiles[sid] = {"reid": feat, "color": color_feat}
                                    th = Thread(
                                        target=save_criminal_data,
                                        args=(args.output_dir, sid, feat, crop.copy(), frame_count),
                                    )
                                    th.start()
                                    threads.append(th)
                                    print(f"🚨 CONFIRMED THREAT SAVED — Suspect #{sid} (Track #{tid}) at frame {frame_count} [{tier}, {live_prob:.0f}%]")
                                else:
                                    print(f"🔗 Re-associated Track #{tid} -> existing Suspect #{sid} at frame {frame_count} [{tier}]")
                    sid_disp = track_to_suspect.get(tid, tid)
                    label = f"SUSPECT #{sid_disp} | {tier} {live_prob:.0f}%"
                    color = (0, 0, 220)
                elif g_hits > 0 or p_hits > 0 or live_prob >= 35.0:
                    label = f"ID #{tid} | ARMING {live_prob:.0f}%"
                    color = (0, 165, 255)
                else:
                    label = f"ID #{tid} | {live_prob:.0f}%"
                    color = (40, 180, 70)

                held_sides = {arm_s for _, arm_s, _ in held_guns}
                active_rays = [g["ray_pts"] for g in arm_grips if has_gun and g["side"] in held_sides]
                cached_people_draw.append((box_np, label, color, is_confirmed, active_rays))

        # Render cached weapon boxes & people annotations smoothly (zero flicker)
        for (wx1, wy1, wx2, wy2, w_conf, w_name), is_held, arm_side in cached_weapon_draw:
            w_col = (0, 0, 255) if is_held else (200, 100, 255)
            cv2.rectangle(annotated, (int(wx1), int(wy1)), (int(wx2), int(wy2)), w_col, 2)
            tag_txt = f"{w_name.upper()} {w_conf:.2f}" + (f" [HELD {arm_side}]" if is_held else "")
            cv2.putText(annotated, tag_txt, (int(wx1), max(15, int(wy1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, w_col, 2)

        active_threats = 0
        for box_np, label, color, is_conf, active_rays in cached_people_draw:
            draw_corner_box(annotated, box_np, color, thickness=3 if is_conf else 2)
            draw_label_pill(annotated, label, int(box_np[0]), int(box_np[1]), color)
            for ray in active_rays:
                if len(ray) >= 2:
                    cv2.line(
                        annotated,
                        (int(ray[0][0]), int(ray[0][1])),
                        (int(ray[-1][0]), int(ray[-1][1])),
                        (0, 255, 255),
                        2,
                    )
                cv2.circle(annotated, (int(ray[0][0]), int(ray[0][1])), 5, (0, 255, 255), -1)
            if is_conf:
                active_threats += 1

        max_threats = max(max_threats, active_threats)
        ft_ms = (time.perf_counter() - t_frame) * 1000.0
        frame_times.append(ft_ms)
        fps_now = 1000.0 / ft_ms if ft_ms > 0 else 0.0

        status_txt = "STATUS: THREAT DETECTED" if active_threats > 0 else "STATUS: MONITORING (CLEAR)"
        status_col = (0, 0, 210) if active_threats > 0 else (35, 160, 65)
        draw_hud_panel(annotated, status_txt, status_col, [
            f"Tracked People: {len(cached_people_draw)}   |   Active Threats: {active_threats}",
            f"Saved Suspects: {len(saved_profiles)}   |   Frame: {frame_count}   |   FPS: {fps_now:.1f}",
        ])

        if writer is not None:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Ground Threat Detector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p") and frame_times:
                print(
                    f"[Perf] Frame: {np.mean(frame_times):.1f}ms | "
                    f"Pose+Track: {np.mean(yolo_times) if yolo_times else 0:.1f}ms | "
                    f"Weapon(Dual): {np.mean(weapon_times) if weapon_times else 0:.1f}ms"
                )

    for th in threads:
        th.join()
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("\n=== Ground Detection Summary ===")
    print(f"Total frames processed : {frame_count}")
    print(f"Max active threats     : {max_threats}")
    print(f"Unique suspects saved  : {len(saved_profiles)} -> {args.output_dir}/")
    if frame_times:
        print(f"Avg frame time / FPS   : {np.mean(frame_times):.1f} ms ({1000.0/np.mean(frame_times):.1f} FPS)")
    if yolo_times:
        print(f"Avg Pose+ByteTrack     : {np.mean(yolo_times):.1f} ms")
    if weapon_times:
        print(f"Avg Weapon (Full+ROI)  : {np.mean(weapon_times):.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground camera ByteTrack + Pose + Dual-Pass Gun-Keypoint Fusion detector")
    parser.add_argument("--source", required=True, help="Video file path or webcam index (e.g. 0)")
    parser.add_argument("--pose-model", default="yolov8s-pose.pt", help="YOLOv8/11-Pose weights (default: yolov8s-pose.pt)")
    parser.add_argument("--weapon-model", default="auto", help="Weapon detector ('auto' CCTV model, 'onnx' Hadi959, 'none', or path)")
    parser.add_argument("--weapon-conf", type=float, default=0.35, help="Weapon confidence threshold (default: 0.35)")
    parser.add_argument("--instant-gun-conf", type=float, default=0.85, help="1-frame instant gun-in-hand confidence (default: 0.85)")
    parser.add_argument("--no-wrist-roi", dest="wrist_roi", action="store_false", help="Disable Pass-2 wrist-ROI zoom inference")
    parser.add_argument("--max-roi-people", type=int, default=3, help="Max people per frame for Wrist-ROI weapon crop (default: 3)")
    parser.add_argument("--osnet-weights", default=None, help="Optional OSNet .pth weights (falls back to MSMT17/ImageNet)")
    parser.add_argument("--output-dir", default="criminal_data", help="Output directory for suspect crops/features")
    parser.add_argument("--frame-skip", type=int, default=2, help="Run inference every Nth frame (default: 2)")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO person detection confidence (default: 0.35)")
    parser.add_argument("--kp-conf", type=float, default=0.55, help="Keypoint confidence threshold (default: 0.55)")
    parser.add_argument("--persistence", type=int, default=8, help="Sliding window size in frames (default: 8)")
    parser.add_argument("--confirm-frames", type=int, default=4, help="Pose-only frames required to confirm (default: 4)")
    parser.add_argument("--gun-confirm", type=int, default=2, help="Gun-in-hand frames required to confirm (default: 2)")
    parser.add_argument("--save-video", default=None, help="Optional path to save annotated output video (.mp4)")
    parser.add_argument("--no-display", action="store_true", help="Run headless without OpenCV GUI window")

    run_detector(parser.parse_args())
