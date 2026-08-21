from pathlib import Path

import numpy as np
from PIL import Image

from gs_video.camera.vipe_worker import _anchor_foreground_exclusion


def _write_image(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def test_anchor_foreground_exclusion_does_not_union_subject_motion(
    tmp_path: Path,
) -> None:
    frames = tuple(tmp_path / "frames" / f"{index:06d}.png" for index in (1, 2))
    masks = tuple(tmp_path / "masks" / f"{index:06d}.png" for index in (1, 2))
    for frame in frames:
        _write_image(frame, np.zeros((10, 10, 3), dtype=np.uint8))
    first = np.zeros((10, 10), dtype=np.uint8)
    first[2:5, 2:5] = 255
    second = np.zeros((10, 10), dtype=np.uint8)
    second[5:8, 5:8] = 255
    _write_image(masks[0], first)
    _write_image(masks[1], second)

    excluded = _anchor_foreground_exclusion(frames, masks)

    assert np.array_equal(excluded, first >= 128)
    assert not excluded[6, 6]
