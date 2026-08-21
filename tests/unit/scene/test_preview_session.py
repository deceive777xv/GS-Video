from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from PIL import Image
import pytest

from gs_video.domain.models import SceneSummary
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.preview_session import PreviewRequestSuperseded, PreviewSession


class QueueStdout:
    def __init__(self) -> None:
        self.lines: deque[bytes] = deque()
        self.closed = False

    def push(self, payload: dict[str, object]) -> None:
        self.lines.append(json.dumps(payload, separators=(",", ":")).encode() + b"\n")

    def readline(self, maximum: int = -1) -> bytes:
        del maximum
        if self.lines:
            return self.lines.popleft()
        return b"" if self.closed else b""

    def close(self) -> None:
        self.closed = True


class ReactiveStdin:
    def __init__(self, process: ScriptedProcess) -> None:
        self.process = process
        self.commands: list[dict[str, object]] = []

    def write(self, payload: bytes) -> int:
        command = json.loads(payload)
        self.commands.append(command)
        output = Path(command["output_path"])
        if command["type"] == "render_live":
            Image.new("RGB", (command["width"], command["height"]), (12, 34, 56)).save(
                output, format="JPEG", quality=85
            )
        else:
            with output.open("xb") as stream:
                np.savez(
                    stream,
                    rgb=np.full(
                        (command["height"], command["width"], 3), 90, dtype=np.uint8
                    ),
                    expected_depth=np.full(
                        (command["height"], command["width"]), 2.0, dtype=np.float32
                    ),
                    opacity=np.ones(
                        (command["height"], command["width"]), dtype=np.float32
                    ),
                )
        self.process.stdout.push(
            {
                "type": "complete",
                "request_id": command["request_id"],
                "output": output.name,
            }
        )
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class BlockingFirstStdin(ReactiveStdin):
    def __init__(self, process: ScriptedProcess) -> None:
        super().__init__(process)
        self.started = threading.Event()
        self.release = threading.Event()
        self._first = True

    def write(self, payload: bytes) -> int:
        if self._first:
            self._first = False
            self.started.set()
            assert self.release.wait(1.0)
        return super().write(payload)


class CrashingStdin:
    def __init__(self, process: ScriptedProcess) -> None:
        self.process = process

    def write(self, payload: bytes) -> int:
        self.process.returncode = 1
        self.process.stdout.close()
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class ScriptedProcess:
    def __init__(self) -> None:
        self.stdout = QueueStdout()
        self.stdout.push(
            {"type": "ready", "implementation_version": "fake-preview-session"}
        )
        self.stderr = io.BytesIO()
        self.stdin = ReactiveStdin(self)
        self.returncode: int | None = None
        self.pid = 4321

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("preview-session", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self.stdout.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()


class RecordingGuard:
    def __init__(self, process: ScriptedProcess) -> None:
        self.process = process
        self.closed = False

    def terminate(self, *, force: bool) -> bool:
        if force:
            self.process.kill()
        else:
            self.process.terminate()
        return True

    def close(self) -> None:
        self.closed = True


def test_preview_session_uses_cache_workspace_for_shared_scene(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    shared_root = tmp_path / "assets"
    shared_root.mkdir()
    scene = shared_root / "scene.ply"
    scene.write_bytes(b"ply\n")
    cache_root = tmp_path / "cache" / "project-1"
    cache_root.mkdir(parents=True)
    summary = SceneSummary(
        filename="scene.ply",
        size=4,
        sha256="a" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    camera = OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=4.0,
        yaw=0.0,
        pitch=0.0,
        fov_y_degrees=60.0,
    )
    process = ScriptedProcess()
    session = PreviewSession(
        worker_prefix=("renderer-python",),
        sh_degree=3,
        available_vram_limit_mb=8192,
        process_factory=lambda *_args, **_options: process,
        tree_guard_factory=RecordingGuard,
    )

    payload = session.render_live(
        project_root,
        scene,
        summary,
        1,
        camera,
        16,
        9,
        preview_root=cache_root,
    )

    assert payload.startswith(b"\xff\xd8")
    output = Path(process.stdin.commands[0]["output_path"])
    assert output.parent == cache_root / "previews"
    assert not (project_root / "source").exists()
    assert not (project_root / "previews").exists()
    session.close()


def test_preview_session_reuses_one_process_for_live_and_authoritative_frames(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source = project_root / "source"
    source.mkdir(parents=True)
    scene = source / "scene.ply"
    scene.write_bytes(b"ply\n")
    (project_root / "previews").mkdir()
    summary = SceneSummary(
        filename="scene.ply",
        size=4,
        sha256="a" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    camera = OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=4.0,
        yaw=0.0,
        pitch=0.0,
        fov_y_degrees=60.0,
    )
    processes: list[ScriptedProcess] = []
    guards: list[RecordingGuard] = []

    def factory(command: list[str], **options: object) -> ScriptedProcess:
        assert "gs_video.scene.preview_worker" in command
        assert options["shell"] is False
        assert options["stdin"] is subprocess.PIPE
        process = ScriptedProcess()
        processes.append(process)
        return process

    def guard_factory(process: ScriptedProcess) -> RecordingGuard:
        guard = RecordingGuard(process)
        guards.append(guard)
        return guard

    session = PreviewSession(
        worker_prefix=("renderer-python",),
        sh_degree=3,
        available_vram_limit_mb=8192,
        process_factory=factory,
        tree_guard_factory=guard_factory,
        log_path=project_root / "logs" / "preview-session.log",
        idle_timeout_seconds=0.05,
    )

    first = session.render_live(
        project_root, "source/scene.ply", summary, 1, camera, 16, 9
    )
    second = session.render_live(
        project_root, "source/scene.ply", summary, 2, camera, 16, 9
    )
    pick = session.render_preview_pick(
        project_root, "source/scene.ply", summary, camera, 16, 9
    )

    assert first.startswith(b"\xff\xd8")
    assert second.startswith(b"\xff\xd8")
    assert pick.rgb.shape == (9, 16, 3)
    assert pick.expected_depth.shape == (9, 16)
    assert len(processes) == 1
    assert [command["type"] for command in processes[0].stdin.commands] == [
        "render_live",
        "render_live",
        "render_pick",
    ]

    deadline = time.monotonic() + 1.0
    while processes[0].returncode is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert processes[0].returncode is not None

    asyncio.run(session.terminate_all())
    assert processes[0].returncode is not None
    assert guards[0].closed is True


def test_preview_session_replaces_pending_live_and_prioritizes_authority(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source = project_root / "source"
    source.mkdir(parents=True)
    scene = source / "scene.ply"
    scene.write_bytes(b"ply\n")
    (project_root / "previews").mkdir()
    summary = SceneSummary(
        filename="scene.ply",
        size=4,
        sha256="a" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    camera = OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=4.0,
        yaw=0.0,
        pitch=0.0,
        fov_y_degrees=60.0,
    )
    process = ScriptedProcess()
    blocking = BlockingFirstStdin(process)
    process.stdin = blocking
    session = PreviewSession(
        worker_prefix=("renderer-python",),
        sh_degree=3,
        available_vram_limit_mb=8192,
        process_factory=lambda *_args, **_options: process,
        tree_guard_factory=RecordingGuard,
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(
            session.render_live,
            project_root,
            "source/scene.ply",
            summary,
            1,
            camera,
            16,
            9,
        )
        assert blocking.started.wait(1.0)
        replaced = pool.submit(
            session.render_live,
            project_root,
            "source/scene.ply",
            summary,
            2,
            camera,
            16,
            9,
        )
        deadline = time.monotonic() + 1.0
        while session._pending_live is None and time.monotonic() < deadline:
            time.sleep(0.005)
        old_ticket = session._pending_live
        newest = pool.submit(
            session.render_live,
            project_root,
            "source/scene.ply",
            summary,
            3,
            camera,
            16,
            9,
        )
        deadline = time.monotonic() + 1.0
        while session._pending_live is old_ticket and time.monotonic() < deadline:
            time.sleep(0.005)
        authority = pool.submit(
            session.render_preview_pick,
            project_root,
            "source/scene.ply",
            summary,
            camera,
            16,
            9,
        )
        deadline = time.monotonic() + 1.0
        while session._authority_waiters == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        blocking.release.set()

        assert first.result(timeout=1).startswith(b"\xff\xd8")
        with pytest.raises(PreviewRequestSuperseded):
            replaced.result(timeout=1)
        assert authority.result(timeout=1).expected_depth.shape == (9, 16)
        assert newest.result(timeout=1).startswith(b"\xff\xd8")

    assert [command["type"] for command in process.stdin.commands] == [
        "render_live",
        "render_pick",
        "render_live",
    ]
    session.close()


def test_preview_session_rebuilds_once_after_worker_crash(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "source"
    source.mkdir(parents=True)
    (project_root / "previews").mkdir()
    (source / "scene.ply").write_bytes(b"ply\n")
    summary = SceneSummary(
        filename="scene.ply",
        size=4,
        sha256="a" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    camera = OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=4.0,
        yaw=0.0,
        pitch=0.0,
        fov_y_degrees=60.0,
    )
    processes: list[ScriptedProcess] = []

    def factory(*_args: object, **_options: object) -> ScriptedProcess:
        process = ScriptedProcess()
        if not processes:
            process.stdin = CrashingStdin(process)  # type: ignore[assignment]
        processes.append(process)
        return process

    session = PreviewSession(
        worker_prefix=("renderer-python",),
        sh_degree=3,
        available_vram_limit_mb=8192,
        process_factory=factory,
        tree_guard_factory=RecordingGuard,
    )

    payload = session.render_live(
        project_root, "source/scene.ply", summary, 1, camera, 16, 9
    )

    assert payload.startswith(b"\xff\xd8")
    assert len(processes) == 2
    session.close()


def test_preview_session_rebuilds_once_after_ready_handshake_crash(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source = project_root / "source"
    source.mkdir(parents=True)
    (project_root / "previews").mkdir()
    (source / "scene.ply").write_bytes(b"ply\n")
    summary = SceneSummary(
        filename="scene.ply",
        size=4,
        sha256="a" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    camera = OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=4.0,
        yaw=0.0,
        pitch=0.0,
        fov_y_degrees=60.0,
    )
    processes: list[ScriptedProcess] = []

    def factory(*_args: object, **_options: object) -> ScriptedProcess:
        process = ScriptedProcess()
        if not processes:
            process.stdout.lines.clear()
            process.stdout.close()
            process.returncode = 1
        processes.append(process)
        return process

    session = PreviewSession(
        worker_prefix=("renderer-python",),
        sh_degree=3,
        available_vram_limit_mb=8192,
        process_factory=factory,
        tree_guard_factory=RecordingGuard,
    )

    payload = session.render_live(
        project_root, "source/scene.ply", summary, 1, camera, 16, 9
    )

    assert payload.startswith(b"\xff\xd8")
    assert len(processes) == 2
    session.close()
