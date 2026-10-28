"""
Generate printable AprilTag images for a range of marker IDs.

Outputs PNG files at high DPI suitable for printing at any size. The
printed physical size depends on the paper you print on; see
docs/02-marker-sizing.md for sizing guidance.

Usage:
    python examples/generate_apriltags.py --ids 1 2 3 4 5 6 7 8 9 10 11
    python examples/generate_apriltags.py --ids 100 101 102 103 104 105
    python examples/generate_apriltags.py --range 1 11 --output-dir markers/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def generate_tag(
    marker_id: int,
    dictionary_id: int = cv2.aruco.DICT_APRILTAG_16h5,
    pixel_size: int = 1000,
    border_bits: int = 1,
    quiet_zone_ratio: float = 0.25,
) -> np.ndarray:
    """
    Generate a single AprilTag image with a white quiet zone border.

    Parameters
    ----------
    marker_id : int
        Marker ID within the dictionary.
    dictionary_id : int
        OpenCV ArUco dictionary constant. Default DICT_APRILTAG_16h5.
    pixel_size : int
        Size of the marker (without quiet zone) in pixels. Bigger = sharper print.
    border_bits : int
        Black border thickness in bit-cell units. AprilTags use 1.
    quiet_zone_ratio : float
        White margin around the tag, as fraction of marker size. 0.25 = 25%
        on each side. Recommended minimum 0.2 for reliable detection.

    Returns
    -------
    image : ndarray (H, W) uint8
        White-on-black tag image with quiet zone, ready to print.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    tag = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixel_size,
                                         borderBits=border_bits)

    margin = int(pixel_size * quiet_zone_ratio)
    canvas = np.full((pixel_size + 2 * margin, pixel_size + 2 * margin),
                     255, dtype=np.uint8)
    canvas[margin:margin + pixel_size, margin:margin + pixel_size] = tag

    # Add ID label below tag
    label = f"ID {marker_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = pixel_size / 1000.0  # scales with tag size
    thickness = max(1, int(font_scale * 2))
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Stack a label area below the canvas
    label_h = int(text_h * 2.5)
    full = np.full((canvas.shape[0] + label_h, canvas.shape[1]),
                   255, dtype=np.uint8)
    full[:canvas.shape[0], :] = canvas

    text_x = (full.shape[1] - text_w) // 2
    text_y = canvas.shape[0] + (label_h + text_h) // 2 - 2
    cv2.putText(full, label, (text_x, text_y), font, font_scale, 0, thickness,
                cv2.LINE_AA)
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate printable AprilTag PNGs.")
    parser.add_argument("--ids", type=int, nargs="+", help="Specific marker IDs.")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                        help="Generate IDs from START to END (inclusive).")
    parser.add_argument("--output-dir", default="markers",
                        help="Output directory for PNGs.")
    parser.add_argument("--pixel-size", type=int, default=1000,
                        help="Marker size in pixels (default 1000).")
    parser.add_argument("--quiet-zone", type=float, default=0.25,
                        help="Quiet zone ratio (default 0.25).")
    args = parser.parse_args()

    if args.ids is None and args.range is None:
        print("ERROR: must provide --ids or --range")
        return 1

    ids = args.ids or list(range(args.range[0], args.range[1] + 1))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(ids)} markers in {out_dir}/")
    for mid in ids:
        try:
            img = generate_tag(mid, pixel_size=args.pixel_size,
                                quiet_zone_ratio=args.quiet_zone)
        except cv2.error as e:
            print(f"  ID {mid}: FAILED ({e})")
            continue
        path = out_dir / f"apriltag_16h5_id{mid:03d}.png"
        cv2.imwrite(str(path), img)
        print(f"  ID {mid}: {path}")

    print("\nPrint these on matte paper at the size needed for your speaker distances.")
    print("See docs/02-marker-sizing.md for sizing guidance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
