"""
Sanity check: verify the install is working end-to-end on synthetic data.

Run this after installing requirements.txt to confirm everything imports
correctly and the calibration solver produces correct results on a
known-truth example.

Usage:
    python examples/sanity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import solve_calibration
from src.geometry import pixel_to_unitvec
from src.pipeline import compute_speaker_direction


def run() -> int:
    print("=== Speaker Array Ground-Truth: install sanity check ===\n")

    # 1. Geometry round-trip
    W, H = 5760, 2880
    v = pixel_to_unitvec((W / 2, H / 2), W, H)
    assert np.allclose(v, [1, 0, 0], atol=1e-9), f"Center pixel should be +x, got {v}"
    print("[OK] Equirectangular geometry round-trip")

    # 2. Calibration solver on synthetic data
    np.random.seed(42)
    true_R = Rotation.from_euler("zyx", [15, -7, 3], degrees=True)
    mic_to_cam = np.array([0.02, 0.0, -0.08])
    calib = {
        100: (3.4, 0.0, 0.0),
        101: (0.0, 2.1, 0.0),
        102: (-2.8, 0.0, 0.0),
        103: (0.0, 0.0, 1.4),
        104: (1.5, 1.5, 0.8),
        105: (1.5, -1.5, -0.3),
    }

    detections = {}
    for mid, xyz in calib.items():
        xyz_cam = np.array(xyz) - mic_to_cam
        v_cam = xyz_cam / np.linalg.norm(xyz_cam)
        v_img = true_R.inv().apply(v_cam)
        az = np.arctan2(v_img[1], v_img[0])
        el = np.arcsin(v_img[2])
        x = (az + np.pi) / (2 * np.pi) * W
        y = (np.pi / 2 - el) / np.pi * H
        detections[mid] = np.array(
            [[x - 3, y - 3], [x + 3, y - 3], [x + 3, y + 3], [x - 3, y + 3]],
            dtype=float,
        )

    R_solved, residual = solve_calibration(
        detections, calib, mic_to_cam, W, H, verbose=False
    )

    delta_deg = float((R_solved * true_R.inv()).magnitude() * 180 / np.pi)
    assert delta_deg < 0.001, f"Recovered R differs by {delta_deg:.4f}°"
    print(f"[OK] Calibration solver recovered ground truth ({delta_deg:.6f}° error)")

    # 3. Speaker direction extraction
    speaker_xyz = (2.0, 1.0, 0.3)
    v_truth = np.array(speaker_xyz) / np.linalg.norm(speaker_xyz)

    # Simulate detection of a speaker at this position
    xyz_cam = np.array(speaker_xyz) - mic_to_cam
    v_cam = xyz_cam / np.linalg.norm(xyz_cam)
    v_img = true_R.inv().apply(v_cam)
    az = np.arctan2(v_img[1], v_img[0])
    el = np.arcsin(v_img[2])
    x = (az + np.pi) / (2 * np.pi) * W
    y = (np.pi / 2 - el) / np.pi * H
    speaker_corners = np.array(
        [[x - 3, y - 3], [x + 3, y - 3], [x + 3, y + 3], [x - 3, y + 3]],
        dtype=float,
    )

    distance = np.linalg.norm(np.array(speaker_xyz) - mic_to_cam)
    v_recovered = compute_speaker_direction(
        speaker_corners, R_solved, mic_to_cam, distance, W, H
    )

    err_deg = float(
        np.degrees(np.arccos(np.clip(v_recovered @ v_truth, -1, 1)))
    )
    assert err_deg < 0.01, f"Speaker direction error {err_deg:.4f}°"
    print(f"[OK] Speaker direction (with parallax correction): {err_deg:.6f}° error")

    print("\nAll checks passed. Install is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
