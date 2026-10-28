#!/usr/bin/env python3
"""calibrate_measure.py — two-mode wrapper around speaker-array-groundtruth.

Modes (selected with --mode):
  calibrate  Solve the rig rotation R from a 360 photo of calibration
             markers. Writes a self-contained calibration YAML carrying
             everything the measurement step needs (R as a 3x3 matrix,
             mic_to_cam, speaker_ids, speaker_distances, residual stats).

  measure    Pulls that YAML + a new 360 photo and writes a
             speaker_positions.json compatible with playback_record.py's
             calibration_manifest input (same schema as run_measurement.py).

Calibration runs ONCE per rig assembly. Measurement runs MANY times — each
photo yields a fresh ground-truth speaker layout while the calibration
parameters (R, mic_to_cam) remain fixed.

Examples
--------
    # Calibrate (uses speaker-array-groundtruth/calibration/calibration_input.yaml):
    python calibrate_measure.py --mode calibrate \
        --config speaker-array-groundtruth/calibration/calibration_input.yaml \
        --output speaker-array-groundtruth/calibration/rig_calibration.yaml

    # Measure (one of many):
    python calibrate_measure.py --mode measure \
        --calibration speaker-array-groundtruth/calibration/rig_calibration.yaml \
        --image  data/scene_001.jpg \
        --output speaker-array-groundtruth/data/speaker_positions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# Make speaker-array-groundtruth importable without installing it.
_SAG_ROOT = Path(__file__).resolve().parent / "speaker-array-groundtruth"
if not _SAG_ROOT.exists():
    sys.exit(f"Cannot find speaker-array-groundtruth at: {_SAG_ROOT}")
sys.path.insert(0, str(_SAG_ROOT))

from scipy.spatial.transform import Rotation  # noqa: E402

import cv2  # noqa: E402

from src.calibration import save_calibration, solve_calibration  # noqa: E402
from src.detection import make_detector  # noqa: E402
from src.geometry import marker_unitvec, unitvec_to_az_el_deg  # noqa: E402
from src.pipeline import compute_speaker_direction  # noqa: E402
from src.validation import leave_one_out  # noqa: E402


# Per-mode config used when --config is not passed (looked up next to this script).
_DEFAULT_CONFIGS = {
    "calibrate": "calibrate_config.yaml",
    "measure":   "measure_config.yaml",
}


# ─────────────────────────── image loading + detection ──────────────────────

def _load_panorama(path: Path) -> np.ndarray:
    """Load an equirectangular panorama as BGR uint8.

    Accepts standard formats via OpenCV (JPG/PNG/TIFF) and DNG via rawpy.
    The DNG MUST already be a stitched equirectangular panorama — raw
    dual-fisheye captures will not work (the pipeline assumes equirectangular
    projection). The Insta360 Studio "Equirectangular DNG" export is fine.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() in (".dng", ".raw", ".arw", ".cr2", ".nef"):
        try:
            import rawpy  # type: ignore
        except ImportError:
            sys.exit(
                f"Reading {path.suffix} requires rawpy:\n"
                f"  pip install rawpy\n"
                f"The DNG must be a stitched equirectangular panorama; raw "
                f"fisheye lens captures are not supported."
            )
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False,
                                  output_bps=8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    return img


def _detect_upscaled_tiles(
    detector,
    img: np.ndarray,
    already_found: set[int],
    scale: int = 3,
    tile: int = 1000,
    overlap: float = 0.35,
) -> dict[int, np.ndarray]:
    """Gap-filling pass: detect markers the full-frame pass missed by scanning
    upscaled tiles of the image.

    A tag on a distant / obliquely-viewed speaker can be too small for the
    detector at native resolution — its black border blends into a dark speaker
    body, so the square contour is never segmented at any threshold. Cropping the
    image into overlapping tiles and upscaling each ~3× (cubic) enlarges those
    tags past the detector's size/contrast floor without the memory blow-up of
    upscaling the whole 7680×3840 panorama at once.

    Only IDs not already in *already_found* are returned; corners are mapped
    back to original-image pixel coordinates (``origin + corner / scale``), so
    they drop straight into the same downstream geometry as the native pass.
    Tiles are full-image (not just the equator) so it recovers off-horizon
    calibration markers too. The seam is already handled upstream, so a marker
    split across the ±180° edge is picked up there, not here.
    """
    H, W = img.shape[:2]
    step = max(1, int(tile * (1.0 - overlap)))
    new: dict[int, np.ndarray] = {}
    for y0 in range(0, H - 1, step):
        for x0 in range(0, W - 1, step):
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            patch = cv2.resize(
                img[y0:y1, x0:x1], None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            corners, ids, _ = detector.detectMarkers(patch)
            if ids is None:
                continue
            for i, mid in enumerate(ids.flatten()):
                mid_int = int(mid)
                if mid_int in already_found or mid_int in new:
                    continue
                cor = corners[i][0].astype(np.float64) / scale
                cor[:, 0] += x0
                cor[:, 1] += y0
                new[mid_int] = cor
    return new


def _detect_with_seam_recovery(
    img: np.ndarray,
    upscale: bool = True,
    upscale_factor: int = 3,
    expected_ids: set[int] | None = None,
) -> tuple[dict[int, np.ndarray], int, int, dict]:
    """Detect AprilTags on an equirectangular image, recovering markers that
    straddle the azimuth ±180° seam, then (optionally) markers too small for the
    full-frame pass via an upscaled-tile gap-filling pass.

    Strategy: detect on the original image, then on a copy rolled W/2 pixels
    horizontally so the seam moves to the image centre. Markers split by the
    original seam are now whole and detectable; their corner coordinates are
    mapped back to the original-image frame. (`marker_unitvec` is periodic in
    azimuth, so split corners average correctly on the unit sphere.)

    If *upscale* is set and some *expected_ids* are still missing after those two
    passes (or *expected_ids* is None), a tiled upscaled pass recovers small /
    low-contrast tags (see :func:`_detect_upscaled_tiles`). It only *adds* markers
    the earlier passes missed, so the easy detections are never disturbed.

    Returns
    -------
    detections : dict[int, ndarray (4, 2)]
    W, H       : int
    recovery   : dict
        {'original_pass_ids': [...], 'rolled_pass_ids': [...],
         'recovered_seam_ids': [...], 'recovered_upscale_ids': [...]}
    """
    H, W = img.shape[:2]
    detector = make_detector()

    detections: dict[int, np.ndarray] = {}

    # Pass 1 — original image.
    corners, ids, _ = detector.detectMarkers(img)
    pass1_ids: list[int] = []
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            detections[int(mid)] = corners[i][0]
            pass1_ids.append(int(mid))

    # Pass 2 — image rolled by W/2 so the seam lands in the middle.
    offset = W // 2
    img_rolled = np.roll(img, offset, axis=1)
    corners_r, ids_r, _ = detector.detectMarkers(img_rolled)
    pass2_ids: list[int] = []
    recovered: list[int] = []
    if ids_r is not None:
        for i, mid in enumerate(ids_r.flatten()):
            mid_int = int(mid)
            pass2_ids.append(mid_int)
            if mid_int in detections:
                continue  # prefer the original-pass detection (no remapping)
            cor = corners_r[i][0].copy().astype(np.float64)
            cor[:, 0] = (cor[:, 0] - offset) % W   # roll back to original frame
            detections[mid_int] = cor
            recovered.append(mid_int)

    # Pass 3 — upscaled-tile gap fill (only if something is still missing).
    upscaled: list[int] = []
    still_missing = (expected_ids is not None
                     and not set(expected_ids).issubset(detections.keys()))
    if upscale and (expected_ids is None or still_missing):
        tile_new = _detect_upscaled_tiles(
            detector, img, set(detections.keys()), scale=int(upscale_factor)
        )
        for mid_int, cor in tile_new.items():
            detections[mid_int] = cor
            upscaled.append(mid_int)

    if not detections:
        raise RuntimeError("No markers detected (after seam + upscale passes).")

    recovery_info = {
        "original_pass_ids":     sorted(pass1_ids),
        "rolled_pass_ids":       sorted(pass2_ids),
        "recovered_seam_ids":    sorted(recovered),
        "recovered_upscale_ids": sorted(upscaled),
    }
    return detections, W, H, recovery_info


def _detect_across_photos(
    image_paths: list[Path],
    upscale: bool = True,
    upscale_factor: int = 3,
    expected_ids: set[int] | None = None,
) -> tuple[dict[int, np.ndarray], int, int, dict]:
    """Detect markers across several photos of the SAME (unmoved) rig and merge.

    ArUco detection is stochastic: a marker clearly visible in the frame can be
    missed in one shot and found in the next. Because the camera pose is
    identical across the supplied photos, every photo observes the same true
    directions, so we can:

      • take the UNION of detected markers  → fills per-shot detection gaps, and
      • AVERAGE the corners of markers seen in more than one photo → reduces
        pixel-level detection noise (improves, never compromises, accuracy).

    A per-marker angular spread across photos is reported so any genuine
    disagreement (e.g. the rig actually moved between shots) is visible rather
    than silently averaged away.

    Returns the same (detections, W, H, info) shape as
    ``_detect_with_seam_recovery`` so it is a drop-in for the single-photo path.
    """
    W = H = None
    corner_stacks: dict[int, list[np.ndarray]] = {}
    photo_hits: dict[int, list[str]] = {}
    per_photo_recovery: dict[str, dict] = {}

    for path in image_paths:
        img = _load_panorama(path)
        det, w, h, rec = _detect_with_seam_recovery(
            img, upscale=upscale, upscale_factor=upscale_factor,
            expected_ids=expected_ids,
        )
        if W is None:
            W, H = w, h
        elif (w, h) != (W, H):
            sys.exit(
                f"Photo dimensions differ ({w}×{h} vs {W}×{H}) for {path}; "
                f"all measurement photos must be the same equirectangular size."
            )
        per_photo_recovery[str(path)] = rec
        extra = ""
        if rec.get("recovered_upscale_ids"):
            extra += f"  [+upscale {rec['recovered_upscale_ids']}]"
        if rec.get("recovered_seam_ids"):
            extra += f"  [+seam {rec['recovered_seam_ids']}]"
        print(f"  {path.name:<32} detected {len(det):>2} markers: "
              f"{sorted(det.keys())}{extra}")
        for mid, cor in det.items():
            corner_stacks.setdefault(mid, []).append(
                np.asarray(cor, dtype=np.float64)
            )
            photo_hits.setdefault(mid, []).append(path.name)

    if W is None:
        raise RuntimeError("No photos supplied for detection.")

    detections: dict[int, np.ndarray] = {}
    spread_deg: dict[int, float] = {}
    for mid, stack in corner_stacks.items():
        detections[mid] = np.mean(np.stack(stack, axis=0), axis=0)
        # Angular agreement across photos (max deviation from the mean direction).
        vecs = [marker_unitvec(c, W, H) for c in stack]
        mean_v = np.mean(vecs, axis=0)
        mean_v = mean_v / np.linalg.norm(mean_v)
        spread_deg[mid] = max(
            (float(np.degrees(np.arccos(
                np.clip(float(np.dot(v, mean_v)), -1.0, 1.0))))
             for v in vecs),
            default=0.0,
        )

    return detections, W, H, {
        "photos":            [str(p) for p in image_paths],
        "per_photo_recovery": per_photo_recovery,
        "photo_hits":        {mid: photo_hits[mid] for mid in sorted(photo_hits)},
        "cross_photo_spread_deg": {
            mid: round(spread_deg[mid], 4) for mid in sorted(spread_deg)
        },
    }


# ─────────────────────────── shared traceback helper ────────────────────────

def _marker_trace(
    label: str,
    detected: list[int],
    expected: list[int],
) -> dict:
    """Print and return a detected-vs-expected breakdown for marker IDs."""
    det_set = set(detected)
    exp_set = set(expected)
    used        = sorted(exp_set & det_set)
    missing     = sorted(exp_set - det_set)
    unexpected  = sorted(det_set - exp_set)

    print(f"\n{label} marker traceback")
    print("-" * (len(label) + 19))
    print(f"  Configured ({len(expected)}): {sorted(expected)}")
    print(f"  Detected   ({len(detected)}): {sorted(detected)}")
    print(f"  Matched    ({len(used)}): {used}")
    if missing:
        print(f"  MISSING    ({len(missing)}): {missing}  "
              f"← configured but not visible in photo")
    else:
        print(f"  MISSING    (0): []")
    if unexpected:
        print(f"  UNEXPECTED ({len(unexpected)}): {unexpected}  "
              f"← detected but not in config (stray tags / wrong layout?)")
    else:
        print(f"  UNEXPECTED (0): []")

    return {
        "detected_marker_ids":   sorted(detected),
        "configured_marker_ids": sorted(expected),
        "matched_marker_ids":    used,
        "missing_marker_ids":    missing,
        "unexpected_marker_ids": unexpected,
    }


# ─────────────────────── calibration residual debug table ───────────────────

def _calibration_residual_table(
    detections: dict[int, np.ndarray],
    calib_markers: dict[int, tuple],
    mic_to_cam: np.ndarray,
    R: "Rotation",
    W: int,
    H: int,
) -> None:
    """Per-marker: detected Cartesian XYZ (camera frame) vs. measured XYZ
    (mic-aligned camera frame), plus angular residual after applying R."""
    mic_to_cam_arr = np.asarray(mic_to_cam, dtype=float)
    matched = sorted(mid for mid in calib_markers if mid in detections)

    print("\nCalibration marker residuals (unit vectors)")
    print(f"  {'ID':>4}  "
          f"{'Detected xyz (cam frame)':^30}  {'Det az':>8}{'Det el':>8}  "
          f"{'Measured xyz (cam-aligned)':^30}  {'Meas az':>8}{'Meas el':>8}  "
          f"{'Δ°':>6}")
    print("  " + "-" * 106)
    for mid in matched:
        xyz_mic = np.asarray(calib_markers[mid], dtype=float)
        v_img = marker_unitvec(detections[mid], W, H)

        xyz_from_cam = xyz_mic - mic_to_cam_arr
        v_cam = xyz_from_cam / np.linalg.norm(xyz_from_cam)

        v_rot = R.apply(v_img)
        dot = float(np.clip(np.dot(v_rot, v_cam), -1.0, 1.0))
        residual_deg = float(np.degrees(np.arccos(dot)))

        az_img, el_img = unitvec_to_az_el_deg(v_img)
        az_cam, el_cam = unitvec_to_az_el_deg(v_cam)

        print(
            f"  {mid:>4}  "
            f"[{v_img[0]:>+7.4f} {v_img[1]:>+7.4f} {v_img[2]:>+7.4f}]  "
            f"{az_img:>+8.2f}{el_img:>+8.2f}  "
            f"[{v_cam[0]:>+7.4f} {v_cam[1]:>+7.4f} {v_cam[2]:>+7.4f}]  "
            f"{az_cam:>+8.2f}{el_cam:>+8.2f}  "
            f"{residual_deg:>6.3f}"
        )


# ─────────────────────────────── calibrate mode ──────────────────────────────

def run_calibrate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Calibration input not found: {config_path}")
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    mic_to_cam = np.array(cfg["mic_to_cam"], dtype=float)
    image_path = cfg["calibration_image"]
    # Marker z may be recorded relative to the ground for convenience; subtract
    # the mic capsule height so calib_markers end up in mic frame (+z up from mic).
    mic_height = float(cfg.get("mic_height", 0.0) or 0.0)
    calib_markers = {
        int(k): (float(v[0]), float(v[1]), float(v[2]) - mic_height)
        for k, v in cfg["calibration_markers"].items()
    }
    speaker_ids = [int(s) for s in cfg.get("speaker_ids", []) or []]
    speaker_distances = {
        int(k): float(v)
        for k, v in (cfg.get("speaker_distances") or {}).items()
    }

    out_yaml = Path(
        args.output
        or cfg.get("output_calibration_yaml")
        or "speaker-array-groundtruth/calibration/rig_calibration.yaml"
    )

    print(f"Image           : {image_path}")
    print(f"Mic→Cam (mic)   : {mic_to_cam.tolist()} m")
    if mic_height:
        print(f"Mic height      : {mic_height} m (subtracted from marker z)")
    print(f"Calib markers   : {len(calib_markers)} configured")

    img = _load_panorama(Path(image_path))
    detections, W, H, recovery = _detect_with_seam_recovery(
        img,
        upscale=bool(cfg.get("upscale_recovery", True)),
        upscale_factor=int(cfg.get("upscale_factor", 3)),
        expected_ids=set(calib_markers.keys()),
    )
    if recovery["recovered_seam_ids"]:
        print(f"Seam-recovered markers: {recovery['recovered_seam_ids']}")
    if recovery["recovered_upscale_ids"]:
        print(f"Upscale-recovered markers: {recovery['recovered_upscale_ids']}")

    trace = _marker_trace(
        label="Calibration",
        detected=list(detections.keys()),
        expected=list(calib_markers.keys()),
    )
    trace["seam_recovery"] = recovery

    R, residual_deg = solve_calibration(
        detections, calib_markers, mic_to_cam, W, H, verbose=True
    )

    _calibration_residual_table(detections, calib_markers, mic_to_cam, R, W, H)

    if not args.skip_loo and len(calib_markers) >= 4:
        print()
        leave_one_out(detections, calib_markers, mic_to_cam, W, H, verbose=True)

    if args.also_save_npy:
        npy_path = Path(args.also_save_npy)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        save_calibration(R, str(npy_path))
        print(f"\nLegacy R .npy   : {npy_path}")

    payload = {
        "metadata": {
            "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
            "calibration_image":  str(image_path),
            "calibration_input":  str(config_path),
            "rms_residual_deg":   round(float(residual_deg), 6),
            "image_width":        int(W),
            "image_height":       int(H),
            "marker_trace":       trace,
        },
        "mic_to_cam":        mic_to_cam.tolist(),
        "R_matrix":          R.as_matrix().tolist(),
        "speaker_ids":       speaker_ids,
        "speaker_distances": speaker_distances,
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"\nCalibration YAML: {out_yaml}")
    print(f"RMS residual     : {residual_deg:.3f}°")
    return 0


# ─────────────────────────────── measure mode ────────────────────────────────

def run_measure(args: argparse.Namespace) -> int:
    # Optional measurement config: any value not given on the CLI is pulled
    # from this YAML (calibration, image, output, speaker_ids, speaker_distances).
    mcfg: dict = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            sys.exit(f"Measurement config not found: {cfg_path}")
        with cfg_path.open() as f:
            mcfg = yaml.safe_load(f) or {}

    calib_path = Path(args.calibration or mcfg.get("calibration") or "")
    if not str(calib_path):
        sys.exit("Missing calibration YAML path (--calibration or 'calibration:' in config).")
    if not calib_path.exists():
        sys.exit(
            f"Calibration YAML not found: {calib_path}\n"
            f"Run --mode calibrate first."
        )
    with calib_path.open() as f:
        calib = yaml.safe_load(f)

    try:
        mic_to_cam = np.array(calib["mic_to_cam"], dtype=float)
        R = Rotation.from_matrix(np.array(calib["R_matrix"], dtype=float))
    except KeyError as exc:
        sys.exit(f"Calibration YAML is missing required field: {exc}")

    # Speaker IDs precedence: CLI > measurement config > calibration YAML.
    speaker_ids_src = (
        args.speaker_ids
        or mcfg.get("speaker_ids")
        or calib.get("speaker_ids")
        or []
    )
    speaker_ids = [int(s) for s in speaker_ids_src]
    if not speaker_ids:
        sys.exit("No speaker_ids in calibration/measurement YAML and none given via --speaker-ids.")

    # Distances precedence: measurement config > calibration YAML.
    speaker_distances = {
        int(k): float(v)
        for k, v in (
            (mcfg.get("speaker_distances")
             if mcfg.get("speaker_distances") is not None
             else calib.get("speaker_distances")) or {}
        ).items()
    }

    # Photo(s): one or many. Multiple photos of the SAME rig are merged to
    # cover markers that a single shot happens to miss (see _detect_across_photos).
    # Accepts a str or a list from either --image (nargs) or the config 'image:'.
    image_src = args.image if args.image else mcfg.get("image")
    if not image_src:
        sys.exit("Missing photo path (--image or 'image:' in config).")
    image_list = image_src if isinstance(image_src, (list, tuple)) else [image_src]
    image_paths = [Path(str(p)) for p in image_list]
    for pth in image_paths:
        if not pth.exists():
            sys.exit(f"Photo not found: {pth}")

    out_str = args.output or mcfg.get("output")
    if not out_str:
        sys.exit("Missing output path (--output or 'output:' in config).")

    print(f"Calibration YAML : {calib_path}")
    if len(image_paths) == 1:
        print(f"Photo            : {image_paths[0]}")
    else:
        print(f"Photos ({len(image_paths)})       : merging detections across shots")
    print(f"Speaker IDs      : {speaker_ids}")
    if speaker_distances:
        print(f"Parallax dist.   : {speaker_distances}")

    upscale_on = bool(mcfg.get("upscale_recovery", True))
    upscale_factor = int(mcfg.get("upscale_factor", 3))
    expected = set(speaker_ids)
    if len(image_paths) == 1:
        img = _load_panorama(image_paths[0])
        detections, W, H, recovery = _detect_with_seam_recovery(
            img, upscale=upscale_on, upscale_factor=upscale_factor,
            expected_ids=expected,
        )
        if recovery["recovered_seam_ids"]:
            print(f"Seam-recovered markers: {recovery['recovered_seam_ids']}")
        if recovery["recovered_upscale_ids"]:
            print(f"Upscale-recovered markers: {recovery['recovered_upscale_ids']}")
    else:
        print("\nPer-photo detection:")
        detections, W, H, recovery = _detect_across_photos(
            image_paths, upscale=upscale_on, upscale_factor=upscale_factor,
            expected_ids=expected,
        )
        spread = recovery["cross_photo_spread_deg"]
        multi = {mid: recovery["photo_hits"][mid]
                 for mid in recovery["photo_hits"]
                 if len(recovery["photo_hits"][mid]) > 1}
        print(f"\nMerged {len(detections)} unique markers across "
              f"{len(image_paths)} photos.")
        # Flag markers whose direction disagrees across photos (rig moved?).
        WARN_DEG = 0.5
        disagree = {mid: spread[mid] for mid in multi if spread.get(mid, 0.0) > WARN_DEG}
        if disagree:
            print(f"  WARNING: {len(disagree)} marker(s) disagree >"
                  f"{WARN_DEG}° across photos — did the rig move between shots?")
            for mid in sorted(disagree):
                print(f"    ID {mid:>3}: {disagree[mid]:.3f}° spread "
                      f"({len(recovery['photo_hits'][mid])} photos)")

    trace = _marker_trace(
        label="Speaker",
        detected=list(detections.keys()),
        expected=speaker_ids,
    )
    trace["seam_recovery"] = recovery

    # Raw detected positions (camera frame, before calibration rotation / parallax)
    detected_sids = [sid for sid in speaker_ids if sid in detections]
    print(f"\nRaw detected positions (camera frame, before R):")
    print(f"  {'ID':>3}  {'xyz (cam frame)':^30}  {'Az (deg)':>9}  {'El (deg)':>9}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*9}  {'-'*9}")
    for sid in sorted(detected_sids):
        v_raw = marker_unitvec(detections[sid], W, H)
        az_raw, el_raw = unitvec_to_az_el_deg(v_raw)
        print(f"  {sid:>3}  [{v_raw[0]:>+7.4f} {v_raw[1]:>+7.4f} {v_raw[2]:>+7.4f}]  "
              f"{az_raw:>+9.3f}  {el_raw:>+9.3f}")

    results: dict[int, tuple[float, float]] = {}
    result_vecs: dict[int, np.ndarray] = {}
    for sid in speaker_ids:
        if sid not in detections:
            continue
        dist = speaker_distances.get(sid)
        v_mic = compute_speaker_direction(
            detections[sid], R, mic_to_cam, dist, W, H
        )
        az, el = unitvec_to_az_el_deg(v_mic)
        results[sid] = (float(az), float(el))
        result_vecs[sid] = v_mic

    print(f"\nSpeaker directions (mic frame):")
    print(f"  {'ID':>3}  {'xyz (mic frame)':^30}  {'Az (deg)':>10}  "
          f"{'El (deg)':>10}  {'Dist (m)':>10}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
    for sid in sorted(results):
        az, el = results[sid]
        v = result_vecs[sid]
        d = speaker_distances.get(sid)
        d_str = f"{d:.2f}" if d is not None else "  -  "
        print(f"  {sid:>3}  [{v[0]:>+7.4f} {v[1]:>+7.4f} {v[2]:>+7.4f}]  "
              f"{az:>+10.3f}  {el:>+10.3f}  {d_str:>10}")

    # Declared but not detected: report each as an error so a missing speaker in
    # the photo is never silently dropped from the measurement run.
    missing_sids = [sid for sid in speaker_ids if sid not in detections]
    if missing_sids:
        print(f"\nERROR: {len(missing_sids)} declared speaker(s) not detected "
              f"in photo — no direction computed:")
        for sid in sorted(missing_sids):
            print(f"  ✗ ID {sid:>3}  declared but not found "
                  f"(occluded, out of frame, or wrong marker ID)")

    out_path = Path(out_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image":         str(image_paths[0]) if len(image_paths) == 1
                         else [str(p) for p in image_paths],
        "calibration":   str(calib_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mic_to_cam":    mic_to_cam.tolist(),
        "image_width":   int(W),
        "image_height":  int(H),
        "marker_trace":  trace,
        "missing_speaker_ids": sorted(missing_sids),
        "speakers": {
            str(sid): {
                "azimuth_deg":   az,
                "elevation_deg": el,
            }
            for sid, (az, el) in results.items()
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSpeaker positions: {out_path}")
    return 0


# ─────────────────────────────────── main ────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Two-mode rig calibration / speaker-direction measurement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode", choices=("calibrate", "measure"), required=True,
        help="Which pipeline to run.",
    )

    # ── calibrate-only ────────────────────────────────────────────────────────
    p.add_argument(
        "--config", default=None,
        help="Input YAML config. Calibrate: mic_to_cam, calibration_image, "
             "calibration_markers, speaker_ids[, speaker_distances]. "
             "Measure: calibration, image, output[, speaker_ids, "
             "speaker_distances]. CLI flags override matching keys. "
             "Defaults to calibrate_config.yaml / measure_config.yaml next to "
             "this script when omitted.",
    )
    p.add_argument(
        "--skip-loo", action="store_true",
        help="(calibrate) Skip leave-one-out cross-validation.",
    )
    p.add_argument(
        "--also-save-npy", default=None, metavar="PATH",
        help="(calibrate) Also write the legacy R .npy at PATH for tools that "
             "still expect it (run_measurement.py, etc.).",
    )

    # ── measure-only ──────────────────────────────────────────────────────────
    p.add_argument(
        "--calibration", default=None,
        help="(measure) Calibration YAML produced by --mode calibrate.",
    )
    p.add_argument(
        "--image", nargs="+", default=None, metavar="PHOTO",
        help="(measure) Equirectangular 360° photo(s) of the speaker layout. "
             "Pass several shots of the SAME rig to merge detections and cover "
             "markers a single shot misses.",
    )
    p.add_argument(
        "--speaker-ids", type=int, nargs="+", default=None, metavar="ID",
        help="(measure) Override speaker IDs to detect in the photo.",
    )

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", default=None,
        help="Output path. Calibrate: .yaml file; measure: .json file.",
    )

    args = p.parse_args()

    # Auto-pick the per-mode config next to this script when --config is omitted,
    # so parameters/paths are pushed from the YAML without repeating the flag.
    if not args.config:
        default_cfg = _DEFAULT_CONFIGS[args.mode]
        candidate = Path(__file__).resolve().parent / default_cfg
        if candidate.exists():
            args.config = str(candidate)
            print(f"Using default {args.mode} config: {candidate}")

    if args.mode == "calibrate":
        if not args.config:
            sys.exit(
                f"--config is required for --mode calibrate "
                f"(no {_DEFAULT_CONFIGS['calibrate']} found next to the script)."
            )
        return run_calibrate(args)

    # measure: each of {calibration, image, output} may come from --config or CLI.
    return run_measure(args)


if __name__ == "__main__":
    sys.exit(main())
