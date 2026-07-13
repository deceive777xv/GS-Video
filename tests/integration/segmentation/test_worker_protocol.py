from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gs_video.domain.errors import UnsupportedMaterialError
from gs_video.segmentation.worker import _has_invisible_run, probe_backend, run_segmentation


class FakePredictor:
    def __init__(self, forward: list[int], reverse: list[int], logits: object) -> None:
        self.forward = forward
        self.reverse = reverse
        self.logits = logits
        self.prompt: dict[str, object] = {}

    def init_state(self, video_path: str) -> dict[str, object]:
        self.video_path = video_path
        return {"num_frames": 3}

    def add_new_points_or_box(self, state: object, **kwargs: object) -> None:
        self.prompt = kwargs

    def propagate_in_video(
        self, state: object, *, start_frame_idx: int, reverse: bool = False
    ) -> object:
        del state, start_frame_idx
        for index in self.reverse if reverse else self.forward:
            yield index, [1], self.logits


class TorchLike:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __getitem__(self, key: object) -> "TorchLike":
        return TorchLike(self.value[key])

    def __gt__(self, threshold: float) -> "TorchLike":
        return TorchLike(self.value > threshold)

    def cpu(self) -> "TorchLike":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


def _assets(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_text("model", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    return config, checkpoint


@pytest.mark.parametrize("backend", ["edgetam", "sam2"])
@pytest.mark.parametrize(
    "logits", [np.ones((1, 1, 4, 5), dtype=np.float32), TorchLike(np.ones((1, 1, 4, 5)))]
)
def test_worker_covers_forward_then_only_missing_reverse_and_preserves_names(
    tmp_path: Path, backend: str, logits: object
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("000002.jpg", "000010.png", "000021.jpg"):
        Image.new("RGB", (5, 4)).save(frames / name)
    predictor = FakePredictor([1, 2], [1, 0], logits)
    events: list[dict[str, object]] = []
    config, checkpoint = _assets(tmp_path)

    result = run_segmentation(
        backend=backend,
        frames_dir=frames,
        output_dir=tmp_path / "masks",
        frame_index=1,
        point=(3, 2),
        config=config,
        checkpoint=checkpoint,
        predictor_factory=lambda *_: predictor,
        emit=events.append,
    )

    assert result == {"type": "result", "mask_dir": "masks", "frames": 3}
    assert [event["current"] for event in events] == [1, 2, 3]
    assert sorted(path.name for path in (tmp_path / "masks").iterdir()) == [
        "000002.png", "000010.png", "000021.png"
    ]
    assert predictor.prompt["frame_idx"] == 1
    assert predictor.prompt["obj_id"] == 1
    np.testing.assert_array_equal(predictor.prompt["points"], [[3.0, 2.0]])
    np.testing.assert_array_equal(predictor.prompt["labels"], [1])


def test_worker_rejects_non_numeric_or_duplicate_frame_stems(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (2, 2)).save(frames / "frame.jpg")
    config, checkpoint = _assets(tmp_path)
    with pytest.raises(ValueError, match="数字"):
        run_segmentation(
            backend="edgetam", frames_dir=frames, output_dir=tmp_path / "masks",
            frame_index=0, point=(1, 1), config=config, checkpoint=checkpoint,
            predictor_factory=lambda *_: FakePredictor([], [], np.ones((1, 1, 2, 2))),
            emit=lambda _: None,
        )


def test_worker_detects_fifteen_consecutive_invisible_masks(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(15):
        Image.new("RGB", (10, 10)).save(frames / f"{index + 1:06d}.jpg")
    predictor = FakePredictor(list(range(15)), [], np.zeros((1, 1, 10, 10)))
    config, checkpoint = _assets(tmp_path)
    with pytest.raises(UnsupportedMaterialError, match="主要人物长时间不可见"):
        run_segmentation(
            backend="edgetam", frames_dir=frames, output_dir=tmp_path / "masks",
            frame_index=0, point=(1, 1), config=config, checkpoint=checkpoint,
            predictor_factory=lambda *_: predictor, emit=lambda _: None,
        )
    assert not (tmp_path / "masks").exists()


def test_probe_reports_exact_backend_config_and_checkpoint(tmp_path: Path) -> None:
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_text("x", encoding="utf-8")
    checkpoint.write_bytes(b"x")
    predictor = FakePredictor([], [], np.ones((1, 1, 1, 1)))
    cleaned: list[bool] = []
    assert probe_backend(
        "sam2", config, checkpoint,
        predictor_factory=lambda actual_config, actual_checkpoint: (
            predictor
            if (actual_config, actual_checkpoint) == (config, checkpoint)
            else pytest.fail("wrong assets")
        ),
        cuda_cleanup=lambda: cleaned.append(True),
    ) == {
        "type": "probe", "backend": "sam2",
        "config": str(config.resolve()), "checkpoint": str(checkpoint.resolve()),
        "predictor": f"{type(predictor).__module__}.{type(predictor).__qualname__}",
    }
    assert cleaned == [True]


def test_probe_failure_still_clears_cuda(tmp_path: Path) -> None:
    config, checkpoint = _assets(tmp_path)
    cleaned: list[bool] = []
    with pytest.raises(RuntimeError, match="build failed"):
        probe_backend(
            "edgetam", config, checkpoint,
            predictor_factory=lambda *_: (_ for _ in ()).throw(RuntimeError("build failed")),
            cuda_cleanup=lambda: cleaned.append(True),
        )
    assert cleaned == [True]


def test_probe_rejects_predictor_without_video_api(tmp_path: Path) -> None:
    config, checkpoint = _assets(tmp_path)
    with pytest.raises(RuntimeError, match="API"):
        probe_backend(
            "edgetam", config, checkpoint, predictor_factory=lambda *_: object(),
            cuda_cleanup=lambda: None,
        )


def test_safe_promotion_rolls_back_old_masks_on_failure(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (4, 4)).save(frames / "000001.jpg")
    output = tmp_path / "masks"
    output.mkdir()
    (output / "old.png").write_bytes(b"old")
    config, checkpoint = _assets(tmp_path)
    predictor = FakePredictor([0], [], np.ones((1, 1, 4, 4)))
    calls = 0

    def replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("promotion failed")
        source.replace(destination)

    with pytest.raises(OSError, match="promotion failed"):
        run_segmentation(
            backend="edgetam", frames_dir=frames, output_dir=output,
            frame_index=0, point=(1, 1), config=config, checkpoint=checkpoint,
            predictor_factory=lambda *_: predictor, emit=lambda _: None,
            replace_path=replace,
        )
    assert (output / "old.png").read_bytes() == b"old"
    assert sorted(path.name for path in output.iterdir()) == ["old.png"]


def test_wrong_size_mask_preserves_old_output_without_debris(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (4, 4)).save(frames / "000001.jpg")
    output = tmp_path / "masks"
    output.mkdir()
    Image.new("L", (4, 4), 255).save(output / "000001.png")
    config, checkpoint = _assets(tmp_path)
    predictor = FakePredictor([0], [], np.ones((1, 1, 2, 2)))

    with pytest.raises(ValueError, match="尺寸"):
        run_segmentation(
            backend="edgetam", frames_dir=frames, output_dir=output,
            frame_index=0, point=(1, 1), config=config, checkpoint=checkpoint,
            predictor_factory=lambda *_: predictor, emit=lambda _: None,
        )

    with Image.open(output / "000001.png") as old_mask:
        assert old_mask.size == (4, 4)
    assert not list(tmp_path.glob(".masks-*"))
    assert not list(tmp_path.glob(".masks.backup-*"))


def test_worker_rejects_boolean_predictor_index(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in (1, 2):
        Image.new("RGB", (2, 2)).save(frames / f"{index:06d}.jpg")
    config, checkpoint = _assets(tmp_path)
    predictor = FakePredictor([True, 0], [], np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError, match="帧索引"):
        run_segmentation(
            backend="sam2", frames_dir=frames, output_dir=tmp_path / "masks",
            frame_index=0, point=(1, 1), config=config, checkpoint=checkpoint,
            predictor_factory=lambda *_: predictor, emit=lambda _: None,
        )


def test_invisibility_run_boundaries() -> None:
    below = 0.0009
    assert not _has_invisible_run([below] * 14)
    assert _has_invisible_run([below] * 15)
    assert not _has_invisible_run([below] * 14 + [0.5] + [below] * 14)
    assert not _has_invisible_run([0.001] * 15)


@pytest.mark.gpu
def test_real_checkpoint_smoke_is_opt_in() -> None:
    pytest.skip("requires a separately installed backend and local checkpoint")
