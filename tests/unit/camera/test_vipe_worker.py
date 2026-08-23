import sys
import types
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from gs_video.camera.vipe_worker import _anchor_foreground_exclusion, _read_depth


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


def test_read_depth_uses_the_openexr_z_channel_without_imageio_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    depth_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    archive_path = tmp_path / "depth.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("00000.exr", b"first")
        archive.writestr("00001.exr", b"second")

    closed: list[bool] = []

    class Coordinate:
        x = 0
        y = 0

    class DataWindow:
        min = Coordinate()
        max = types.SimpleNamespace(x=1, y=1)

    class InputFile:
        def __init__(self, stream) -> None:
            assert stream.read() == b"first"

        def header(self) -> dict[str, object]:
            return {"dataWindow": DataWindow()}

        def channels(self, names: list[str]) -> list[bytes]:
            assert names == ["Z"]
            return [depth_values.tobytes()]

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setitem(sys.modules, "OpenEXR", types.SimpleNamespace(InputFile=InputFile))

    depth, names = _read_depth(archive_path, 2)

    assert names == ("00000.exr", "00001.exr")
    assert depth.dtype == np.float64
    assert np.array_equal(depth, depth_values.astype(np.float64))
    assert closed == [True]
