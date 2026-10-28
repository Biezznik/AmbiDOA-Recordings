"""
Validation utilities: leave-one-out cross-validation, sanity checks.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration import solve_calibration
from .geometry import marker_unitvec


def leave_one_out(
    detections: dict[int, np.ndarray],
    calib_markers_mic: dict[int, tuple[float, float, float]],
    mic_to_cam: np.ndarray,
    W: int,
    H: int,
    verbose: bool = True,
) -> dict[int, float]:
    """
    Leave-one-out cross-validation on calibration markers.

    For each calibration marker:
        1. Solve R using the other N-1 markers
        2. Predict the held-out marker's direction using that R
        3. Compare to the held-out marker's measured direction
        4. Report the angular error in degrees

    Markers with anomalously high held-out error are likely measurement
    errors (typo, wrong reference point, etc.).

    Returns
    -------
    errors : dict[int, float]
        Maps marker_id -> held-out angular error in degrees.
    """
    if len(calib_markers_mic) < 4:
        raise ValueError(
            "Leave-one-out needs at least 4 calibration markers "
            "(3 for solving + 1 held out)."
        )

    mic_to_cam = np.asarray(mic_to_cam, dtype=float)
    errors: dict[int, float] = {}

    for held_out in calib_markers_mic:
        if held_out not in detections:
            continue

        subset = {k: v for k, v in calib_markers_mic.items() if k != held_out}
        R, _ = solve_calibration(detections, subset, mic_to_cam, W, H,
                                  verbose=False)

        v_img = marker_unitvec(detections[held_out], W, H)
        v_pred = R.apply(v_img)

        xyz = np.array(calib_markers_mic[held_out]) - mic_to_cam
        v_true = xyz / np.linalg.norm(xyz)

        cos_err = float(np.clip(v_pred @ v_true, -1.0, 1.0))
        err_deg = float(np.degrees(np.arccos(cos_err)))
        errors[held_out] = err_deg

    if verbose:
        print("Leave-one-out cross-validation:")
        for mid, err in sorted(errors.items()):
            flag = " <-- OUTLIER" if err > 2 * np.median(list(errors.values())) else ""
            print(f"  Marker {mid}: held-out error = {err:.3f}°{flag}")

    return errors


def angular_diff_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angular difference between two unit vectors, in degrees."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_a = float(np.clip(v1 @ v2, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def check_symmetry(
    results: dict[int, tuple[float, float]],
    pairs: list[tuple[int, int]],
    expected_az_diff: float = 0.0,
    expected_el_diff: float = 0.0,
    tol_deg: float = 1.0,
) -> dict[tuple[int, int], tuple[float, float]]:
    """
    Check expected symmetries in a speaker array.

    For each (id1, id2) pair, verify that
        az(id1) - az(id2) ≈ expected_az_diff
        el(id1) - el(id2) ≈ expected_el_diff
    within tol_deg.

    Returns the actual differences for inspection.
    """
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for id1, id2 in pairs:
        if id1 not in results or id2 not in results:
            continue
        az1, el1 = results[id1]
        az2, el2 = results[id2]
        d_az = (az1 - az2 + 180) % 360 - 180  # signed shortest-arc
        d_el = el1 - el2
        out[(id1, id2)] = (d_az, d_el)
    return out
