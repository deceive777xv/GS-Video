"""Controlled camera motion solving and trajectory mapping."""

from gs_video.camera.classify import CameraKind
from gs_video.camera.mapping import map_ground_aligned_trajectory
from gs_video.camera.solution import CameraSolution, SourceGroundEstimate

__all__ = [
    "CameraKind",
    "CameraSolution",
    "SourceGroundEstimate",
    "map_ground_aligned_trajectory",
]
