import numpy as np

from gs_video.scene.worker import _MatrixCamera


def test_matrix_camera_scales_authoritative_intrinsics_for_preview_size() -> None:
    camera = _MatrixCamera(
        np.eye(4),
        60.0,
        np.array(
            [[800.0, 0.0, 600.0], [0.0, 810.0, 330.0], [0.0, 0.0, 1.0]]
        ),
        (1280, 720),
    )

    np.testing.assert_allclose(
        camera.intrinsics(640, 360),
        np.array(
            [[400.0, 0.0, 300.0], [0.0, 405.0, 165.0], [0.0, 0.0, 1.0]]
        ),
    )
