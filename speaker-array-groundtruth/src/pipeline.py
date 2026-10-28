"""
End-to-end pipeline for extracting speaker directions from a 360° photo.

Given a saved calibration rotation R and a known camera-mic offset, this
module turns each detected speaker marker into an (azimuth, elevation)
in the mic frame.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .detection import detect_markers
from .geometry import marker_unitvec, unitvec_to_az_el_deg


def compute_speaker_direction(
    corners4: np.ndarray,
    R: Rotation,
    mic_to_cam: np.ndarray,
    distance_m: float | None,
    W: int,
    H: int,
) -> np.ndarray:
    """
    Compute a speaker's direction in the mic frame from its detected marker.

    Parameters
    ----------
    corners4 : ndarray (4, 2)
        Marker corner pixel coordinates.
    R : Rotation
        Calibration rotation (image frame -> mic-aligned axes).
    mic_to_cam : ndarray (3,)
        Vector from mic to camera, in mic frame.
    distance_m : float or None
        Distance from camera to speaker in meters. If provided, applies
        parallax correction to give the direction from MIC, not from CAMERA.
        If None, returns the direction from CAMERA but expressed in
        mic-aligned axes — accurate to within (offset / distance) radians.
    W, H : int
        Image dimensions.

    Returns
    -------
    v_mic : ndarray (3,)
        Unit vector pointing from mic (or camera, if distance_m is None) to
        the speaker, in mic-frame coordinates.
    """
    v_img = marker_unitvec(corners4, W, H)            # camera frame
    v_aligned = R.apply(v_img)                        # mic-aligned axes, origin = camera

    if distance_m is None:
        return v_aligned

    # Reconstruct marker position in mic frame, then take direction from mic.
    p_marker_mic = np.asarray(mic_to_cam, dtype=float) + distance_m * v_aligned
    return p_marker_mic / np.linalg.norm(p_marker_mic)


def process_photo(
    image_path: str,
    R: Rotation,
    mic_to_cam: np.ndarray,
    speaker_ids: list[int],
    speaker_distances: dict[int, float] | None = None,
    verbose: bool = True,
) -> dict[int, tuple[float, float]]:
    """
    Full pipeline: detect markers in a photo, compute speaker directions.

    Parameters
    ----------
    image_path : str
        Path to equirectangular 360° image.
    R : Rotation
        Calibration rotation, previously solved via solve_calibration().
    mic_to_cam : ndarray (3,)
        Camera-mic offset in mic frame.
    speaker_ids : list[int]
        Marker IDs to treat as speakers.
    speaker_distances : dict[int, float] or None
        Optional per-speaker distance in meters for parallax correction.
        Speakers not in this dict are processed without parallax correction.
    verbose : bool
        Print results if True.

    Returns
    -------
    results : dict[int, (azimuth_deg, elevation_deg)]
    """
    detections, W, H = detect_markers(image_path)
    speaker_distances = speaker_distances or {}

    if verbose:
        print(f"Detected {len(detections)} markers in {image_path}")

    results: dict[int, tuple[float, float]] = {}
    missing: list[int] = []

    for sid in speaker_ids:
        if sid not in detections:
            missing.append(sid)
            continue
        dist = speaker_distances.get(sid)
        v_mic = compute_speaker_direction(
            detections[sid], R, mic_to_cam, dist, W, H
        )
        results[sid] = unitvec_to_az_el_deg(v_mic)

    if verbose:
        print(f"\nSpeaker directions (mic frame):")
        print(f"  {'ID':>3}  {'Az (deg)':>10}  {'El (deg)':>10}  {'Dist (m)':>10}")
        print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}")
        for sid in sorted(results):
            az, el = results[sid]
            d = speaker_distances.get(sid)
            d_str = f"{d:.2f}" if d is not None else "  -  "
            print(f"  {sid:>3}  {az:>+10.3f}  {el:>+10.3f}  {d_str:>10}")
        if missing:
            print(f"\nNot detected: {missing}")

    return results
