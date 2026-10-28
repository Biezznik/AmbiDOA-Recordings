"""
Calibration: solve the rotation R that aligns the camera frame to the mic frame.

The Insta360 camera and the ambisonic mic are mounted rigidly together, but
their coordinate frames differ by a rotation (and a small translation, the
camera-mic offset).

Calibration markers are placed at known (x, y, z) positions in the MIC frame.
For each calibration marker, we have:

    v_img : unit vector from CAMERA to marker (read from image pixels)
    v_cam : unit vector from CAMERA to marker (computed as
            (xyz_from_mic - mic_to_cam) / norm), i.e. the marker position
            relative to the camera, normalized.

These two should be identical after rotating v_img by R. We solve R via
Kabsch (scipy's Rotation.align_vectors) over all calibration markers,
which is a least-squares solver that handles the overdetermined case
gracefully.

The translation `mic_to_cam` is needed to convert mic-frame measurements
into camera-frame direction vectors. It must be measured separately (with
calipers, once per rig assembly).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import marker_unitvec


def solve_calibration(
    detections: dict[int, np.ndarray],
    calib_markers_mic: dict[int, tuple[float, float, float]],
    mic_to_cam: np.ndarray,
    W: int,
    H: int,
    verbose: bool = True,
) -> tuple[Rotation, float]:
    """
    Solve for rotation R that maps camera-frame image directions onto
    mic-aligned camera-frame measured directions.

    Parameters
    ----------
    detections : dict[int, ndarray]
        Output of detect_markers(). Maps marker_id -> (4,2) corner array.
    calib_markers_mic : dict[int, (x,y,z)]
        Maps calibration marker IDs -> position in MIC frame (meters).
    mic_to_cam : ndarray of shape (3,)
        Vector from mic capsule to camera optical center, in MIC frame.
    W, H : int
        Image dimensions.
    verbose : bool
        Print summary if True.

    Returns
    -------
    R : scipy Rotation
        Rotation aligning image-frame vectors to mic-aligned axes.
    rmsd_deg : float
        RMS angular residual of the calibration fit, in degrees.
    """
    if len(calib_markers_mic) < 3:
        raise ValueError("Need at least 3 calibration markers.")

    img_vecs: list[np.ndarray] = []
    cam_vecs: list[np.ndarray] = []
    used_ids: list[int] = []
    missing_ids: list[int] = []

    mic_to_cam = np.asarray(mic_to_cam, dtype=float)

    for mid, xyz_from_mic in calib_markers_mic.items():
        if mid not in detections:
            missing_ids.append(mid)
            continue

        # Direction from camera to marker, from image pixels (camera frame)
        v_img = marker_unitvec(detections[mid], W, H)

        # Direction from camera to marker, from measurement (mic-aligned axes)
        xyz_from_cam = np.asarray(xyz_from_mic, dtype=float) - mic_to_cam
        norm = np.linalg.norm(xyz_from_cam)
        if norm < 1e-9:
            raise ValueError(
                f"Marker {mid} is at the camera origin (after offset). "
                "Place calibration markers further away."
            )
        v_cam = xyz_from_cam / norm

        img_vecs.append(v_img)
        cam_vecs.append(v_cam)
        used_ids.append(mid)

    if len(img_vecs) < 3:
        raise RuntimeError(
            f"Only {len(img_vecs)} calibration markers detected and measured; "
            "need at least 3."
        )

    img_vecs_arr = np.array(img_vecs)
    cam_vecs_arr = np.array(cam_vecs)

    R, rmsd = Rotation.align_vectors(cam_vecs_arr, img_vecs_arr)
    rmsd_deg = float(np.degrees(rmsd))

    if verbose:
        print(f"Calibration: used {len(used_ids)} markers (IDs {used_ids})")
        if missing_ids:
            print(f"  Missing (not detected): {missing_ids}")
        print(f"  RMS angular residual: {rmsd_deg:.3f}°")
        if rmsd_deg > 5:
            print("  WARNING: residual > 0.5°. Check measurements for typos "
                  "or marker placement errors.")

    return R, rmsd_deg


def save_calibration(R: Rotation, path: str) -> None:
    """Save a Rotation as a .npy file containing its rotation matrix."""
    np.save(path, R.as_matrix())


def load_calibration(path: str) -> Rotation:
    """Load a Rotation previously saved by save_calibration."""
    return Rotation.from_matrix(np.load(path))
