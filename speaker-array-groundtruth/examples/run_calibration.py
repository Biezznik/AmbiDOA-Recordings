"""
Run once per session: solve and save the rig calibration rotation R.

Usage:
    python examples/run_calibration.py
    python examples/run_calibration.py --config calibration/calibration_input.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import detect_markers
from src.calibration import solve_calibration, save_calibration
from src.validation import leave_one_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve and save rig calibration R.")
    parser.add_argument(
        "--config",
        default="calibration/calibration_input.yaml",
        help="Path to calibration YAML.",
    )
    parser.add_argument(
        "--skip-loo",
        action="store_true",
        help="Skip leave-one-out cross-validation.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        return 1

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    mic_to_cam = np.array(cfg["mic_to_cam"], dtype=float)
    image_path = cfg["calibration_image"]
    calib_markers = {int(k): tuple(v) for k, v in cfg["calibration_markers"].items()}
    output_path = Path(cfg["output_calibration"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading image: {image_path}")
    print(f"Camera-mic offset (mic frame): {mic_to_cam} m")
    print(f"Calibration markers: {len(calib_markers)} configured")
    print()

    detections, W, H = detect_markers(image_path)
    print(f"Detected {len(detections)} markers: {sorted(detections.keys())}")
    print()

    R, residual_deg = solve_calibration(
        detections, calib_markers, mic_to_cam, W, H, verbose=True
    )

    if not args.skip_loo and len(calib_markers) >= 4:
        print()
        leave_one_out(detections, calib_markers, mic_to_cam, W, H, verbose=True)

    save_calibration(R, str(output_path))
    print(f"\nSaved calibration to: {output_path}")
    print(f"Final RMS residual: {residual_deg:.3f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
