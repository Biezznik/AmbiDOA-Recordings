"""
AprilTag detection on equirectangular images.

Uses OpenCV's ArUco module with the AprilTag detector and
APRILTAG-specific corner refinement for sub-pixel accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np


# Default dictionary: AprilTag 16h5
# - 4x4 data grid: large bit cells, good for long range
# - 30 unique markers: plenty for 11 speakers + ~10 calibration markers
# - Higher minimum Hamming distance than equivalent ArUco => fewer false positives
DEFAULT_DICTIONARY = cv2.aruco.DICT_APRILTAG_16h5


def make_detector(
    dictionary_id: int = DEFAULT_DICTIONARY,
    refine_window: int = 5,
    max_iterations: int = 50,
    min_accuracy: float = 0.05,
) -> cv2.aruco.ArucoDetector:
    """
    Construct an ArucoDetector configured for AprilTag detection with
    sub-pixel corner refinement.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    params.cornerRefinementWinSize = refine_window
    params.cornerRefinementMaxIterations = max_iterations
    params.cornerRefinementMinAccuracy = min_accuracy
    # ── Robustness for 360° equirectangular photos ────────────────────────────
    # Tags on distant speakers are small in-frame, and equirectangular projection
    # bends their (straight) edges into slight curves — most pronounced away from
    # the equator. The stock ArUco defaults reject both cases. Loosen the two
    # purely-geometric gates so smaller and mildly-curved quads survive to the
    # ID-decoding stage; decoding itself is left strict (default error
    # correction) so false positives stay near zero with the low-Hamming 16h5.
    params.minMarkerPerimeterRate = 0.01        # was 0.03 → accept smaller tags
    params.polygonalApproxAccuracyRate = 0.06   # was 0.03 → tolerate curved edges
    params.adaptiveThreshWinSizeMax = 53        # was 23 → wider threshold sweep
    params.adaptiveThreshWinSizeStep = 4        # was 10 → finer threshold sweep
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_markers(
    image_path: str,
    dictionary_id: int = DEFAULT_DICTIONARY,
) -> tuple[dict[int, np.ndarray], int, int]:
    """
    Detect markers in an equirectangular image.

    Returns
    -------
    detections : dict[int, ndarray]
        Mapping marker_id -> array of shape (4, 2) of corner pixel coords.
    W, H : int
        Image width and height in pixels.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]

    detector = make_detector(dictionary_id)
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) == 0:
        raise RuntimeError(f"No markers detected in {image_path}")

    detections = {int(ids[i][0]): corners[i][0] for i in range(len(ids))}
    return detections, W, H


def detect_markers_from_array(
    img: np.ndarray,
    dictionary_id: int = DEFAULT_DICTIONARY,
) -> tuple[dict[int, np.ndarray], int, int]:
    """Same as detect_markers, but takes an already-loaded image array."""
    H, W = img.shape[:2]
    detector = make_detector(dictionary_id)
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) == 0:
        raise RuntimeError("No markers detected.")

    detections = {int(ids[i][0]): corners[i][0] for i in range(len(ids))}
    return detections, W, H
