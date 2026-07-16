import numpy as np
import pytest
from pydantic import ValidationError

from gs_video.api.schemas import ApiError, CameraInput
from gs_video.api.workflow import validate_pick_buffer
from gs_video.domain.contracts import PickBuffer
from gs_video.domain.models import CameraPose, VideoSummary


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_persisted_and_request_camera_values_must_be_finite(value: float) -> None:
    camera = {
        "target": [0.0, 0.0, 0.0],
        "distance": 4.0,
        "yaw": value,
        "pitch": 0.0,
        "fov_y_degrees": 55.0,
    }

    with pytest.raises(ValidationError):
        CameraPose.model_validate(camera)
    with pytest.raises(ValidationError):
        CameraInput.model_validate(camera)


def test_video_summary_duration_must_be_finite() -> None:
    with pytest.raises(ValidationError):
        VideoSummary(
            filename="clip.mp4",
            size=1,
            sha256="a" * 64,
            width=1920,
            height=1080,
            duration_seconds=float("inf"),
            fps="30",
            has_audio=True,
            frame_count=300,
        )


@pytest.mark.parametrize(
    "buffer",
    [
        PickBuffer(
            rgb=np.zeros((9, 16, 4), dtype=np.uint8),
            expected_depth=np.ones((9, 16), dtype=np.float32),
        ),
        PickBuffer(
            rgb=np.zeros((9, 16, 3), dtype=np.uint8),
            expected_depth=np.ones((8, 16), dtype=np.float32),
        ),
        PickBuffer(
            rgb=np.zeros((9, 16, 3), dtype=np.uint8),
            expected_depth=np.full((9, 16), np.nan, dtype=np.float32),
        ),
    ],
)
def test_preview_renderer_output_is_validated_before_publication(
    buffer: PickBuffer,
) -> None:
    with pytest.raises(ApiError) as caught:
        validate_pick_buffer(buffer, width=16, height=9)

    assert caught.value.envelope.code == "invalid_preview"
