"""
Geometric primitives: pixel <-> unit vector <-> spherical conversions.

Convention (AmbiX / SN3D):
    +x = forward
    +y = left
    +z = up
    azimuth = CCW from +x in the xy-plane
    elevation = up from the xy-plane

Equirectangular pixel mapping. Pixel-x increases rightward in PHYSICAL
space (the standard 360-camera unwrap, verified on Qoocam 8K Enterprise),
so azimuth — CCW from +x in AmbiX — DECREASES as pixel-x grows:
    azimuth   = pi - (x / W) * 2*pi    # x = W/2 → az = 0 (forward),
                                       # x = 3W/4 → az = -pi/2 (right),
                                       # x = W/4  → az = +pi/2 (left)
    elevation = pi/2 - (y / H) * pi
"""
from __future__ import annotations

import numpy as np


def pixel_to_unitvec(px: tuple[float, float], W: int, H: int) -> np.ndarray:
    """Equirectangular pixel -> unit direction vector (camera frame)."""
    x, y = px
    az = np.pi - (x / W) * 2 * np.pi
    el = np.pi / 2 - (y / H) * np.pi
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)])


def marker_unitvec(corners4: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Marker direction as the mean of corner directions on the unit sphere.

    More accurate than averaging pixels first under equirectangular projection,
    especially for markers placed off the equator (|elevation| > 20°).

    Parameters
    ----------
    corners4 : ndarray of shape (4, 2)
        Pixel coordinates of the four marker corners.
    """
    vecs = np.array([pixel_to_unitvec(c, W, H) for c in corners4])
    mean_vec = vecs.mean(axis=0)
    return mean_vec / np.linalg.norm(mean_vec)


def unitvec_to_az_el_deg(v: np.ndarray) -> tuple[float, float]:
    """Unit vector (3,) -> (azimuth_deg, elevation_deg)."""
    az = np.degrees(np.arctan2(v[1], v[0]))
    el = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return float(az), float(el)


def az_el_to_unitvec(az_rad: float, el_rad: float) -> np.ndarray:
    """(azimuth, elevation) in radians -> unit vector."""
    return np.array([np.cos(el_rad) * np.cos(az_rad),
                     np.cos(el_rad) * np.sin(az_rad),
                     np.sin(el_rad)])


def cartesian_to_unitvec(xyz: np.ndarray) -> np.ndarray:
    """Cartesian point -> unit direction vector (from origin)."""
    v = np.asarray(xyz, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / n
