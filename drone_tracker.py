"""
Aerial Drone Suspect Tracker (ByteTrack + Centroid/Gallery Re-ID + Unlock Patience)
=================================================================================
Loads suspect Re-ID embeddings from `criminal_data/`, tracks people using
YOLOv8s + ByteTrack (90-frame buffer), fuses MSMT17 Centroid + Max-View FAISS
matching with 2D Outfit-Color verification, and sustains suspect lock across
brief occlusions via locked-track querying, unlock patience, and spatial re-attach.

Usage:
    python drone_tracker.py --source path/to/drone_video.mp4
    python drone_tracker.py --source 0 --no-display --save-video drone_out.mp4
"""

import os
import glob
import json
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from reid_utils import (
    load_osnet_model,
    extract_reid_feature,
    extract_outfit_color_feature,
    create_faiss_index,
    draw_hud_panel,
    draw_label_pill,
    draw_corner_box,
)

MAX_IMAGES_PER_CRIMINAL = 7


class MultiImageTracker:
    """Manages centroid-anchored multi-image Re-ID galleries and FAISS index."""

    def __init__(self, data_dir: str = "criminal_data", max_images: int = MAX_IMAGES_PER_CRIMINAL):
        self.data_dir = data_dir
        self.max_images = max_images
        self.tracking_data = {}
        self.faiss_index = create_faiss_index(512)
        self.all_person_ids = []
        self.all_image_indices = []

    def load_saved_criminal_data(self) -> int:
        feat_files = sorted(glob.glob(os.path.join(self.data_dir, "features", "*.npy")))
        img_dir = os.path.join(self.data_dir, "images")
        loaded = 0

        for fpath in feat_files:
            base = os.path.splitext(os.path.basename(fpath))[0]
            parts = base.split("_")
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue

            feat = np.load(fpath).astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            img_path = os.path.join(img_dir, f"{base}.jpg")
            img = cv2.imread(img_path) if os.path.isfile(img_path) else np.zeros((128, 64, 3), dtype=np.uint8)
            color_feat = extract_outfit_color_feature(img)

            if pid not in self.tracking_data:
                self.tracking_data[pid] = {
                    "stored_features": [feat],
                    "stored_colors": [color_feat],
                    "stored_images": [img],
                    "slot_scores": [1.0],  # Slot #0 is pinned ground-camera anchor
                    "centroid": feat.copy(),
                    "centroid_color": color_feat.copy(),
                }
            elif len(self.tracking_data[pid]["stored_features"]) < self.max_images:
                self.tracking_data[pid]["stored_features"].append(feat)
                self.tracking_data[pid]["stored_colors"].append(color_feat)
                self.tracking_data[pid]["stored_images"].append(img)
                self.tracking_data[pid]["slot_scores"].append(0.85)
                self._update_centroid(pid)
            loaded += 1

        self.rebuild_index()
        print(f"Loaded {len(self.tracking_data)} suspect profile(s) ({loaded} embeddings) from '{self.data_dir}/'.")
        return len(self.tracking_data)

    def _update_centroid(self, pid: int):
        """Anchor centroid (50% slot #0 ground threat + 50% mean of verified views)."""
        data = self.tracking_data[pid]
        feats = np.vstack(data["stored_features"])
        cols = np.vstack(data["stored_colors"])
        mean_f = np.mean(feats, axis=0)
        mean_c = np.mean(cols, axis=0)
        cent_f = 0.50 * data["stored_features"][0] + 0.50 * mean_f
        cent_c = 0.50 * data["stored_colors"][0] + 0.50 * mean_c
        data["centroid"] = (cent_f / (np.linalg.norm(cent_f) + 1e-8)).astype(np.float32)
        data["centroid_color"] = (cent_c / (np.linalg.norm(cent_c) + 1e-8)).astype(np.float32)

    def rebuild_index(self):
        features_list, pids, img_idxs = [], [], []
        for pid, data in self.tracking_data.items():
            for idx, feat in enumerate(data["stored_features"]):
                features_list.append(feat)
                pids.append(pid)
                img_idxs.append(idx)

        self.faiss_index.reset()
        self.all_person_ids = pids
        self.all_image_indices = img_idxs
        if features_list:
            self.faiss_index.add(np.vstack(features_list).astype(np.float32))

    def _fuse_scores(self, reid_sim: float, color_sim: float) -> float:
        if color_sim < 0.35:
            return 0.35 * reid_sim + 0.65 * color_sim
        return 0.68 * reid_sim + 0.32 * color_sim

    def query_suspect_score(self, pid: int, query_feature: np.ndarray, query_color: np.ndarray):
        """
        Score a query specifically against suspect `pid` using:
        45% centroid similarity + 55% best gallery-view similarity.
        Returns (fused_sim, reid_sim, color_sim, best_view_idx).
        """
        data = self.tracking_data.get(pid)
        if data is None or query_feature is None:
            return 0.0, 0.0, 0.0, 0

        feats = np.vstack(data["stored_features"])
        cols = np.vstack(data["stored_colors"])
        reid_sims = feats @ query_feature
        col_sims = cols @ query_color

        best_idx = 0
        best_view_fused = -1.0
        for idx, (rs, cs) in enumerate(zip(reid_sims, col_sims)):
            vf = self._fuse_scores(float(rs), float(cs))
            if vf > best_view_fused:
                best_view_fused = vf
                best_idx = idx

        cent_reid = float(np.dot(data["centroid"], query_feature))
        cent_col = float(np.dot(data["centroid_color"], query_color))
        cent_fused = self._fuse_scores(cent_reid, cent_col)

        fused = 0.45 * cent_fused + 0.55 * best_view_fused
        best_reid = 0.45 * cent_reid + 0.55 * float(reid_sims[best_idx])
        best_col = 0.45 * cent_col + 0.55 * float(col_sims[best_idx])
        return fused, best_reid, best_col, best_idx

    def query_best_candidate(self, query_feature: np.ndarray, query_color: np.ndarray):
        """Return (suspect_id, fused_sim, reid_sim, color_sim, image_index) across all suspects."""
        if query_feature is None or self.faiss_index.ntotal == 0:
            return None, 0.0, 0.0, 0.0, 0
        q = query_feature.reshape(1, -1).astype(np.float32)
        k = min(10, self.faiss_index.ntotal)
        _, idxs = self.faiss_index.search(q, k)

        checked_pids = set()
        best = (None, 0.0, 0.0, 0.0, 0)
        for flat_idx in idxs[0]:
            flat_idx = int(flat_idx)
            if flat_idx < 0:
                continue
            pid = self.all_person_ids[flat_idx]
            if pid in checked_pids:
                continue
            checked_pids.add(pid)
            fused, rs, cs, img_idx = self.query_suspect_score(pid, query_feature, query_color)
            if fused > best[1]:
                best = (pid, fused, rs, cs, img_idx)
        return best

    def update_gallery(
        self,
        pid: int,
        new_image: np.ndarray,
        new_feature: np.ndarray,
        new_color: np.ndarray,
        frame_idx: int,
        fused_sim: float,
        reid_sim: float,
        color_sim: float,
    ):
        entry = self.tracking_data[pid]
        if fused_sim < 0.72 or reid_sim < 0.65 or color_sim < 0.60:
            return

        existing = np.vstack(entry["stored_features"])
        max_existing_sim = float(np.max(existing @ new_feature))
        if max_existing_sim >= 0.88:
            return

        if len(entry["stored_features"]) < self.max_images:
            img_idx = len(entry["stored_features"])
            entry["stored_features"].append(new_feature)
            entry["stored_colors"].append(new_color)
            entry["stored_images"].append(new_image.copy())
            entry["slot_scores"].append(fused_sim)
            self._update_centroid(pid)
            self._save_update(pid, new_feature, new_image, frame_idx, img_idx)
            self.rebuild_index()
            print(f"🔄 Suspect #{pid}: Added diverse view {img_idx + 1}/{self.max_images} (fused={fused_sim:.3f}, reid={reid_sim:.3f}, col={color_sim:.3f})")
        elif fused_sim >= 0.78 and len(entry["slot_scores"]) > 1:
            # Quality-aware replacement: keep slot #0 pinned, replace weakest non-pinned slot
            weak_idx = 1 + int(np.argmin(entry["slot_scores"][1:]))
            if fused_sim > entry["slot_scores"][weak_idx] + 0.04:
                entry["stored_features"][weak_idx] = new_feature
                entry["stored_colors"][weak_idx] = new_color
                entry["stored_images"][weak_idx] = new_image.copy()
                entry["slot_scores"][weak_idx] = fused_sim
                self._update_centroid(pid)
                self._save_update(pid, new_feature, new_image, frame_idx, weak_idx)
                self.rebuild_index()

    def _save_update(self, pid: int, feat: np.ndarray, img: np.ndarray, frame_idx: int, img_idx: int):
        img_dir = os.path.join(self.data_dir, "images")
        feat_dir = os.path.join(self.data_dir, "features")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(feat_dir, exist_ok=True)
        tag = f"criminal_{pid}_0_frame_{frame_idx}_updated_{img_idx}"
        cv2.imwrite(os.path.join(img_dir, f"{tag}.jpg"), img)
        np.save(os.path.join(feat_dir, f"{tag}.npy"), feat)

        meta_path = os.path.join(self.data_dir, f"tracking_metadata_{pid}.json")
        with open(meta_path, "w") as f:
            json.dump({
                "suspect_id": pid,
                "total_images": len(self.tracking_data[pid]["stored_images"]),
                "last_updated_frame": frame_idx,
            }, f, indent=2)


def draw_suspect_thumbnail(frame: np.ndarray, bbox, ref_img: np.ndarray):
    if ref_img is None or ref_img.size == 0:
        return
    thumb = cv2.resize(ref_img, (48, 96))
    fh, fw = frame.shape[:2]
    x2, y1 = int(bbox[2]), int(bbox[1])
    tx1 = max(2, min(fw - 52, x2 + 8))
    ty1 = max(2, min(fh - 100, y1))
    if tx1 + 48 <= fw and ty1 + 96 <= fh:
        cv2.rectangle(frame, (tx1 - 2, ty1 - 2), (tx1 + 50, ty1 + 98), (0, 0, 220), 2)
        frame[ty1:ty1 + 96, tx1:tx1 + 48] = thumb


def _box_center_dist(b1, b2) -> float:
    c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
    c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def run_tracker(args):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open drone video source: {args.source}")

    yolo_model = YOLO(args.yolo_model)
    reid_model, device = load_osnet_model(args.osnet_weights)

    tracker_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack_custom.yaml")
    if not os.path.isfile(tracker_cfg):
        tracker_cfg = "bytetrack.yaml"

    tracker = MultiImageTracker(data_dir=args.data_dir)
    n_profiles = tracker.load_saved_criminal_data()
    if n_profiles == 0:
        print(f"⚠️  No suspect profiles found in '{args.data_dir}/'. Run detector.py first.")

    writer = None
    if args.save_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not args.no_display:
        cv2.namedWindow("Aerial Suspect Tracker", cv2.WINDOW_NORMAL)

    track_locks = {}      # tid -> {"suspect_id": int|None, "ema_sim": float, "ref_idx": int, "miss_streak": int}
    last_suspect_pos = {} # pid -> (box, frame_idx) for spatial re-attach after occlusion
    tracking_history = defaultdict(list)
    cached_draw = []

    frame_count = 0
    max_suspects = 0
    yolo_times, reid_times, frame_times = [], [], []

    print(
        f"Starting aerial tracking (ByteTrack-90 + Centroid/Gallery + Patience={args.unlock_patience}) "
        f"on: {args.source} (high={args.threshold:.2f}, low={args.low_threshold:.2f})"
    )

    while cap.isOpened():
        t_frame = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()

        if frame_count % args.frame_skip == 0:
            t0 = time.perf_counter()
            results = yolo_model.track(
                frame, persist=True, tracker=tracker_cfg, conf=args.conf, classes=[0], verbose=False
            )
            yolo_times.append((time.perf_counter() - t0) * 1000.0)

            cached_draw = []
            used_suspects_in_frame = set()

            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.asarray(r.boxes.xyxy)
                tids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else list(range(len(boxes)))

                for tid, box in zip(tids, boxes):
                    x1, y1, x2, y2 = map(int, box)
                    if x2 - x1 < 30 or y2 - y1 < 50:
                        continue

                    h_f, w_f = frame.shape[:2]
                    crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                    if crop.size == 0:
                        continue

                    t_r = time.perf_counter()
                    feat = extract_reid_feature(crop, reid_model, device)
                    color_feat = extract_outfit_color_feature(crop)
                    reid_times.append((time.perf_counter() - t_r) * 1000.0)

                    state = track_locks.get(tid, {"suspect_id": None, "ema_sim": 0.0, "ref_idx": 0, "miss_streak": 0})

                    if state["suspect_id"] is None:
                        cand_pid, raw_sim, reid_sim, color_sim, ref_idx = tracker.query_best_candidate(feat, color_feat)
                        ema = raw_sim if state["ema_sim"] == 0.0 else (0.55 * raw_sim + 0.45 * state["ema_sim"])
                        state["ema_sim"] = ema

                        # Spatial + Re-ID quick re-attach if near recently seen suspect position
                        spatial_ok = False
                        if cand_pid is not None and cand_pid in last_suspect_pos:
                            prev_box, prev_f = last_suspect_pos[cand_pid]
                            if (frame_count - prev_f) <= 30 and _box_center_dist(box, prev_box) <= 120.0:
                                spatial_ok = (raw_sim >= args.low_threshold and color_sim >= 0.45)

                        if cand_pid is not None and cand_pid not in used_suspects_in_frame and color_sim >= 0.40:
                            if ema >= args.threshold or spatial_ok:
                                state["suspect_id"] = cand_pid
                                state["ref_idx"] = ref_idx
                                state["miss_streak"] = 0
                                tag = "RE-ATTACH" if spatial_ok and ema < args.threshold else "LOCKED"
                                print(f"🚨 {tag} SUSPECT #{cand_pid} on Track #{tid} (EMA={ema:.3f}, ReID={reid_sim:.3f}, Col={color_sim:.3f}) at frame {frame_count}")
                    else:
                        # Continuous locked-track query specifically against locked_pid's gallery
                        locked_pid = state["suspect_id"]
                        raw_sim, reid_sim, color_sim, ref_idx = tracker.query_suspect_score(locked_pid, feat, color_feat)
                        ema = 0.45 * raw_sim + 0.55 * state["ema_sim"]
                        state["ema_sim"] = ema
                        state["ref_idx"] = ref_idx

                        if ema < args.low_threshold:
                            state["miss_streak"] += 1
                            if state["miss_streak"] >= args.unlock_patience:
                                print(f"🔓 Released Track #{tid} from Suspect #{locked_pid} after {state['miss_streak']} low frames")
                                state["suspect_id"] = None
                                state["miss_streak"] = 0
                        else:
                            state["miss_streak"] = 0

                    track_locks[tid] = state
                    locked_pid = state["suspect_id"]

                    if locked_pid is not None and locked_pid not in used_suspects_in_frame:
                        used_suspects_in_frame.add(locked_pid)
                        last_suspect_pos[locked_pid] = (box.copy(), frame_count)
                        ref_img = tracker.tracking_data[locked_pid]["stored_images"][state["ref_idx"]]
                        tracker.update_gallery(
                            locked_pid, crop, feat, color_feat, frame_count,
                            raw_sim, reid_sim, color_sim,
                        )
                        tracking_history[locked_pid].append((frame_count, state["ema_sim"]))
                        cached_draw.append((box, f"SUSPECT #{locked_pid} | {state['ema_sim']*100:.0f}%", (0, 0, 220), True, ref_img))
                    else:
                        pct = state["ema_sim"] * 100.0
                        if state["ema_sim"] >= args.low_threshold and color_sim >= 0.40:
                            cached_draw.append((box, f"Person #{tid} | NEAR {pct:.0f}%", (0, 165, 255), False, None))
                        else:
                            cached_draw.append((box, f"Person #{tid} | {pct:.0f}%", (40, 180, 70), False, None))

        suspects_now = 0
        for box, label, color, is_suspect, ref_img in cached_draw:
            draw_corner_box(annotated, box, color, thickness=3 if is_suspect else 2)
            draw_label_pill(annotated, label, int(box[0]), int(box[1]), color)
            if is_suspect:
                suspects_now += 1
                draw_suspect_thumbnail(annotated, box, ref_img)

        max_suspects = max(max_suspects, suspects_now)
        ft_ms = (time.perf_counter() - t_frame) * 1000.0
        frame_times.append(ft_ms)
        fps_now = 1000.0 / ft_ms if ft_ms > 0 else 0.0

        status_txt = "STATUS: SUSPECT LOCKED" if suspects_now > 0 else "STATUS: AERIAL SCANNING"
        status_col = (0, 0, 210) if suspects_now > 0 else (35, 160, 65)
        draw_hud_panel(annotated, status_txt, status_col, [
            f"People: {len(cached_draw)}   |   Suspects Locked: {suspects_now}",
            f"Thresh (H/L): {args.threshold:.2f}/{args.low_threshold:.2f}   |   Frame: {frame_count}   |   FPS: {fps_now:.1f}",
        ])

        if writer is not None:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Aerial Suspect Tracker", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p") and frame_times:
                print(
                    f"[Perf] Frame: {np.mean(frame_times):.1f}ms | "
                    f"YOLO+Track: {np.mean(yolo_times) if yolo_times else 0:.1f}ms | "
                    f"ReID: {np.mean(reid_times) if reid_times else 0:.1f}ms"
                )

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("\n=== Aerial Tracking Summary ===")
    print(f"Total frames read       : {frame_count}")
    print(f"Max suspects in a frame : {max_suspects}")
    print(f"Unique suspects tracked : {len(tracking_history)}")
    if frame_times:
        print(f"Avg frame time / FPS    : {np.mean(frame_times):.1f} ms ({1000.0/np.mean(frame_times):.1f} FPS)")
    for pid, hist in tracking_history.items():
        sims = [s for _, s in hist]
        print(f"  Suspect #{pid}: {len(hist)} tracked windows, Max sim={max(sims):.3f}, Avg sim={np.mean(sims):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial drone ByteTrack + centroid/gallery FAISS suspect tracker")
    parser.add_argument("--source", required=True, help="Drone video path or webcam index (e.g. 0)")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="YOLOv8 weights (default: yolov8s.pt)")
    parser.add_argument("--osnet-weights", default=None, help="Optional OSNet .pth weights (falls back to MSMT17/ImageNet)")
    parser.add_argument("--data-dir", default="criminal_data", help="Directory with saved suspect embeddings")
    parser.add_argument("--threshold", type=float, default=0.60, help="High lock threshold (default: 0.60)")
    parser.add_argument("--low-threshold", type=float, default=0.48, help="Low sustain hysteresis threshold (default: 0.48)")
    parser.add_argument("--unlock-patience", type=int, default=6, help="Consecutive low frames before releasing lock (default: 6)")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame (default: 2)")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO person confidence (default: 0.40)")
    parser.add_argument("--save-video", default=None, help="Optional path to save annotated output video (.mp4)")
    parser.add_argument("--no-display", action="store_true", help="Run headless without OpenCV GUI window")

    run_tracker(parser.parse_args())
