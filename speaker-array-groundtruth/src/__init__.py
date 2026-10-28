"""Speaker array ground-truth measurement using Insta360 + AprilTags."""

from .geometry import pixel_to_unitvec, marker_unitvec, unitvec_to_az_el_deg
from .detection import detect_markers
from .calibration import solve_calibration
from .pipeline import compute_speaker_direction, process_photo
from .validation import leave_one_out

__all__ = [
    "pixel_to_unitvec",
    "marker_unitvec",
    "unitvec_to_az_el_deg",
    "detect_markers",
    "solve_calibration",
    "compute_speaker_direction",
    "process_photo",
    "leave_one_out",
]
