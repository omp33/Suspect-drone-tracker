"""
Unit tests for Suspect-drone-tracker geometry, firearm attribution,
pose threat rules, outfit color separation, and checkpoint security checks.
Runs offline without downloading model weights.
"""

import unittest
import zipfile
import pickle
import numpy as np

from detector import (
    get_arm_grips,
    build_wrist_roi,
    deduplicate_weapons,
    assign_weapons_to_people,
    analyze_pose_threat,
)
from reid_utils import (
    extract_outfit_color_feature,
    verify_pytorch_checkpoint_safety,
)


def _blank_keypoints():
    return np.zeros((17, 3), dtype=np.float32)


class TestGunKeypointFusion(unittest.TestCase):
    def test_forearm_grip_attribution_picks_holding_person(self):
        # Shooter on left extending right arm toward x=245, y=180
        kp_shooter = _blank_keypoints()
        kp_shooter[6] = [160, 175, 0.95]   # R shoulder
        kp_shooter[8] = [200, 178, 0.95]   # R elbow
        kp_shooter[10] = [235, 180, 0.95]  # R wrist
        kp_shooter[12] = [160, 290, 0.95]  # R hip
        box_shooter = np.array([120, 120, 250, 420], dtype=np.float32)

        # Bystander far away
        kp_bystander = _blank_keypoints()
        kp_bystander[6] = [480, 175, 0.95]
        kp_bystander[8] = [480, 225, 0.95]
        kp_bystander[10] = [480, 275, 0.95]
        box_bystander = np.array([450, 120, 560, 420], dtype=np.float32)

        entries = [
            (1, box_shooter, get_arm_grips(kp_shooter, box_shooter)),
            (2, box_bystander, get_arm_grips(kp_bystander, box_bystander)),
        ]
        # Pistol right along shooter's projected hand/barrel
        pistol = [(242.0, 172.0, 272.0, 196.0, 0.78, "weapon")]
        assigned = assign_weapons_to_people(pistol, entries)

        self.assertIn(1, assigned)
        self.assertNotIn(2, assigned)
        self.assertEqual(assigned[1][0][1], "R")

    def test_oversized_false_weapon_rejected(self):
        kp = _blank_keypoints()
        kp[8] = [150, 200, 0.95]
        kp[10] = [160, 230, 0.95]
        box = np.array([100, 100, 200, 350], dtype=np.float32)  # 100x250 = 25000 px^2
        entries = [(1, box, get_arm_grips(kp, box))]
        # False torso/crate box (> 14% person area)
        huge_box = [(110.0, 150.0, 195.0, 280.0, 0.80, "weapon")]
        assigned = assign_weapons_to_people(huge_box, entries)
        self.assertEqual(len(assigned), 0)

    def test_deduplicate_weapons_iou_nms(self):
        dets = [
            (100.0, 100.0, 130.0, 130.0, 0.45, "weapon"),
            (102.0, 101.0, 131.0, 129.0, 0.74, "weapon"),
            (300.0, 200.0, 325.0, 225.0, 0.61, "weapon"),
        ]
        kept = deduplicate_weapons(dets, iou_thresh=0.45)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0][4], 0.74)

    def test_wrist_roi_bounds_inside_frame(self):
        kp = _blank_keypoints()
        kp[7] = [30, 40, 0.9]
        kp[9] = [15, 25, 0.9]
        box = np.array([10, 10, 90, 210], dtype=np.float32)
        roi = build_wrist_roi(kp, box, frame_w=640, frame_h=480)
        self.assertIsNotNone(roi)
        rx1, ry1, rx2, ry2 = roi
        self.assertGreaterEqual(rx1, 0)
        self.assertGreaterEqual(ry1, 0)
        self.assertLessEqual(rx2, 640)
        self.assertLessEqual(ry2, 480)


class TestPoseAndSecurity(unittest.TestCase):
    def test_side_horizontal_aim_detected_and_idle_rejected(self):
        # Extended horizontal aiming arm
        kp_aim = _blank_keypoints()
        kp_aim[5] = [100, 150, 0.95]
        kp_aim[6] = [130, 150, 0.95]
        kp_aim[8] = [195, 152, 0.95]
        kp_aim[10] = [255, 154, 0.95]
        kp_aim[12] = [130, 280, 0.95]
        box = np.array([90, 100, 260, 420], dtype=np.float32)
        res_aim = analyze_pose_threat(kp_aim, box)
        self.assertTrue(res_aim["is_threat"])
        self.assertIn("R", res_aim["semi_ready_arms"])

        # Natural arms hanging down at sides
        kp_idle = _blank_keypoints()
        kp_idle[5] = [100, 150, 0.95]
        kp_idle[6] = [150, 150, 0.95]
        kp_idle[7] = [95, 210, 0.95]
        kp_idle[8] = [155, 210, 0.95]
        kp_idle[9] = [92, 275, 0.95]
        kp_idle[10] = [158, 275, 0.95]
        kp_idle[11] = [105, 280, 0.95]
        kp_idle[12] = [145, 280, 0.95]
        res_idle = analyze_pose_threat(kp_idle, np.array([85, 100, 165, 420], dtype=np.float32))
        self.assertFalse(res_idle["is_threat"])

    def test_outfit_color_separates_black_and_white(self):
        black_img = np.full((128, 64, 3), 20, dtype=np.uint8)
        white_img = np.full((128, 64, 3), 235, dtype=np.uint8)
        f_blk = extract_outfit_color_feature(black_img)
        f_wht = extract_outfit_color_feature(white_img)
        self.assertLess(float(np.dot(f_blk, f_wht)), 0.40)

    def test_checkpoint_safety_blocks_unsafe_pickle_globals(self):
        import os
        import tempfile

        class MaliciousPayload:
            def __reduce__(self):
                return (os.system, ("echo blocked",))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            tmp_path = tf.name
        try:
            with zipfile.ZipFile(tmp_path, "w") as zf:
                zf.writestr("archive/data.pkl", pickle.dumps(MaliciousPayload()))
            self.assertFalse(verify_pytorch_checkpoint_safety(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
