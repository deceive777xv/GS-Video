import stat
import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from gs_video.api.schemas import ApiError, CameraInput
from gs_video.api.export_routes import _read_stable_artifact
from gs_video.api.workflow import (
    PreviewArtifactStore,
    PreviewCoordinator,
    GsplatPreviewService,
    validate_pick_buffer,
)
from gs_video.domain.contracts import PickBuffer
from gs_video.domain.models import CameraPose, SceneSummary, VideoSummary
from gs_video.scene.camera import OrbitCamera


class _FakeArtifactHandle:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._read = False

    def __enter__(self) -> "_FakeArtifactHandle":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def fileno(self) -> int:
        return 42

    def read(self, size: int) -> bytes:
        del size
        if self._read:
            return b""
        self._read = True
        return self._payload


class _FakeArtifactPath:
    def __init__(self, payload: bytes, path_stat: object) -> None:
        self._payload = payload
        self._path_stat = path_stat

    def open(self, mode: str) -> _FakeArtifactHandle:
        assert mode == "rb"
        return _FakeArtifactHandle(self._payload)

    def stat(self) -> object:
        return self._path_stat


def _artifact_stat(*, inode: int = 10, size: int = 4) -> object:
    return SimpleNamespace(
        st_dev=1,
        st_ino=inode,
        st_size=size,
        st_nlink=1,
        st_mtime_ns=100,
        st_ctime_ns=100,
        st_mode=stat.S_IFREG,
    )


def test_stable_artifact_reader_rejects_path_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _artifact_stat()
    path = _FakeArtifactPath(b"test", _artifact_stat(inode=11))
    monkeypatch.setattr("gs_video.api.routes.os.fstat", lambda _fd: expected)

    with pytest.raises(ApiError) as caught:
        _read_stable_artifact(path, expected, limit=4)  # type: ignore[arg-type]

    assert caught.value.envelope.code == "artifact_changed"


def test_stable_artifact_reader_stops_at_the_captured_size_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _artifact_stat()
    path = _FakeArtifactPath(b"oversized", expected)
    monkeypatch.setattr("gs_video.api.routes.os.fstat", lambda _fd: expected)

    with pytest.raises(ApiError) as caught:
        _read_stable_artifact(path, expected, limit=4)  # type: ignore[arg-type]

    assert caught.value.envelope.code == "artifact_changed"


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


def test_preview_coordinator_skips_obsolete_queued_generations() -> None:
    async def exercise() -> None:
        coordinator = PreviewCoordinator()
        started = Event()
        release = Event()
        rendered: list[int] = []

        def render(generation: int) -> int:
            rendered.append(generation)
            if generation == 1:
                started.set()
                assert release.wait(2)
            return generation

        first = asyncio.create_task(coordinator.render(1, render, 1))
        assert await asyncio.to_thread(started.wait, 1)
        second = asyncio.create_task(coordinator.render(2, render, 2))
        third = asyncio.create_task(coordinator.render(3, render, 3))
        await asyncio.sleep(0)
        release.set()
        outcomes = await asyncio.gather(first, second, third, return_exceptions=True)

        assert rendered == [1, 3]
        assert isinstance(outcomes[0], ApiError)
        assert isinstance(outcomes[1], ApiError)
        assert outcomes[2] == 3

    asyncio.run(exercise())


def test_preview_cancellation_keeps_serialization_until_thread_finishes() -> None:
    async def exercise() -> None:
        coordinator = PreviewCoordinator()
        started = Event()
        release = Event()
        second_started = Event()

        def first_render() -> int:
            started.set()
            assert release.wait(2)
            return 1

        def second_render() -> int:
            second_started.set()
            return 2

        first = asyncio.create_task(coordinator.render(1, first_render))
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        second = asyncio.create_task(coordinator.render(2, second_render))
        await asyncio.sleep(0.05)
        assert not second_started.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == 2

    asyncio.run(exercise())


def test_preview_artifacts_are_bounded_while_preserving_authoritative_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    (root / "previews").mkdir(parents=True)
    store = PreviewArtifactStore(root, buffer_limit=2)
    buffer = PickBuffer(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        expected_depth=np.ones((2, 2), dtype=np.float32),
    )

    first, size, digest = store.publish(buffer)
    store.publish(buffer)
    third, _, _ = store.publish(buffer, preserve_artifact_ids={first})

    assert store.read(
        artifact_id=first, expected_size=size, expected_sha256=digest
    )
    assert store.pick_buffer(third) is buffer
    assert len(list((root / "previews").glob("*.png"))) == 2


def test_preview_scene_cache_rejects_bytes_that_do_not_match_import_summary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    source = root / "source"
    source.mkdir(parents=True)
    scene = source / "scene.ply"
    scene.write_bytes(b"changed-scene")
    service = GsplatPreviewService()

    with pytest.raises(ApiError) as caught:
        service.render_pick(
            root,
            "source/scene.ply",
            SceneSummary(
                filename="scene.ply",
                size=len(b"changed-scene"),
                sha256="0" * 64,
                gaussian_count=1,
                estimated_vram_mb=1,
            ),
            OrbitCamera(
                target=(0.0, 0.0, 0.0),
                distance=4.0,
                yaw=0.0,
                pitch=0.0,
                fov_y_degrees=60.0,
            ),
            16,
            9,
        )

    assert caught.value.envelope.code == "scene_changed"


def test_preview_coordinator_coalesces_duplicate_queued_generation() -> None:
    async def exercise() -> None:
        coordinator = PreviewCoordinator()
        started = Event()
        release = Event()
        rendered: list[str] = []

        def render(label: str) -> str:
            rendered.append(label)
            if label == "first":
                started.set()
                assert release.wait(2)
            return label

        first = asyncio.create_task(coordinator.render(1, render, "first"))
        assert await asyncio.to_thread(started.wait, 1)
        latest = asyncio.create_task(coordinator.render(1, render, "latest"))
        await asyncio.sleep(0)
        release.set()
        outcomes = await asyncio.gather(first, latest, return_exceptions=True)

        assert rendered == ["first"]
        assert outcomes == ["first", "first"]

    asyncio.run(exercise())
