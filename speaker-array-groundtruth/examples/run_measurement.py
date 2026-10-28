"""
Run once per photo: detect speaker markers, output (azimuth, elevation).

Uses the saved calibration from run_calibration.py. Calibration markers do
NOT need to be visible in the photo — only the speaker markers.

Usage:
    python examples/run_measurement.py --image data/scene_001.jpg
    python examples/run_measurement.py --image data/scene_001.jpg --output data/scene_001.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import load_calibration
from src.pipeline import process_photo


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute speaker (az, el) from a 360 photo.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to equirectangular 360° photo.",
    )
    parser.add_argument(
        "--config",
        default="calibration/calibration_input.yaml",
        help="Calibration YAML (provides mic_to_cam, speaker_ids, etc.).",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to saved R .npy. Defaults to value in config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for results.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mic_to_cam = np.array(cfg["mic_to_cam"], dtype=float)
    speaker_ids = list(cfg["speaker_ids"])
    speaker_distances = cfg.get("speaker_distances") or {}
    speaker_distances = {int(k): float(v) for k, v in speaker_distances.items()}

    calib_path = args.calibration or cfg["output_calibration"]
    if not Path(calib_path).exists():
        print(f"ERROR: calibration file not found: {calib_path}")
        print("Run run_calibration.py first to create it.")
        return 1

    R = load_calibration(calib_path)
    print(f"Loaded calibration from: {calib_path}")
    print()

    results = process_photo(
        args.image,
        R,
        mic_to_cam,
        speaker_ids,
        speaker_distances=speaker_distances,
        verbose=True,
    )

    if args.output:
        out = {
            "image": args.image,
            "calibration": calib_path,
            "mic_to_cam": mic_to_cam.tolist(),
            "speakers": {
                str(sid): {"azimuth_deg": az, "elevation_deg": el}
                for sid, (az, el) in results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
