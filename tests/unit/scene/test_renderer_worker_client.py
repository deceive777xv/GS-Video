from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from gs_video.camera.serialization import (
    MappedTrajectory,
    read_mapped_trajectory,
    write_mapped_trajectory,
)
from gs_video.domain.errors import CancelledError, GsVideoError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.scene.camera import OrbitCamera
import gs_video.scene.worker_client as worker_client_module
from gs_video.scene.worker_client import RendererWorkerClient
from gs_video.scene.worker_protocol import (
    OrbitCameraPayload,
    ProbeRequest,
    RenderPickRequest,
    RenderSequenceRequest,
)


class FakeProcess:
    def __init__(self, lines: list[str], *, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = io.StringIO("".join(lines))
        self.stderr = io.StringIO(stderr)
        self.returncode: int | None = returncode
        self.pid = 1234
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class BlockingProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__([])
        self.returncode = None

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("renderer", timeout)
        return self.returncode


class RecordingGuard:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.calls: list[bool] = []
        self.closed = False

    def terminate(self, *, force: bool) -> bool:
        self.calls.append(force)
        self.process.returncode = -9 if force else -15
        return True

    def close(self) -> None:
        self.closed = True


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    scene = (tmp_path / "scene.ply").absolute()
    scene.write_bytes(b"ply\n")
    manifest = (tmp_path / "trajectory.json").absolute()
    write_mapped_trajectory(
        manifest,
        MappedTrajectory(
            fov_y_degrees=60.0,
            camera_to_world=(OrbitCamera(
                target=(0.0, 0.0, 1.0), distance=1.0, yaw=0.0,
                pitch=0.0, fov_y_degrees=60.0,
            ).camera_to_world(),),
        ),
    )
    return scene, manifest


def _sequence_request(tmp_path: Path) -> RenderSequenceRequest:
    scene, manifest = _inputs(tmp_path)
    return RenderSequenceRequest(
        type="render_sequence",
        scene_path=scene,
        camera_manifest=manifest,
        output_dir=(tmp_path / "frames").absolute(),
        width=64,
        height=36,
        sh_degree=1,
        background=(0.0, 0.0, 0.0),
        preview_stride=1,
    )


def _client(
    tmp_path: Path,
    factory: Any,
    *,
    guard_factory: Any | None = None,
) -> RendererWorkerClient:
    return RendererWorkerClient(
        worker_prefix=("renderer-python",),
        process_factory=factory,
        tree_guard_factory=guard_factory or (lambda process: RecordingGuard(process)),
        log_path=tmp_path / "logs" / "renderer-worker.log",
    )


def test_render_client_sends_argv_without_shell_or_token_and_validates_inventory(
    tmp_path: Path,
) -> None:
    request = _sequence_request(tmp_path)
    process = FakeProcess([
        json.dumps({"type": "progress", "current": 1, "total": 1, "message": "one"}) + "\n",
        json.dumps({
            "type": "complete", "implementation_version": "gsplat-1.5.3",
            "outputs": ["000001.png"],
        }) + "\n",
    ])
    observed: dict[str, object] = {}

    def factory(command: list[str], **options: object) -> FakeProcess:
        observed["command"] = command
        observed["options"] = options
        request.output_dir.mkdir()
        Image.new("RGB", (64, 36), (1, 2, 3)).save(request.output_dir / "000001.png")
        return process

    events: list[tuple[int, int, str]] = []
    result = _client(tmp_path, factory).render_sequence(
        request, lambda *event: events.append(event), CancellationToken()
    )

    command = observed["command"]
    options = observed["options"]
    assert isinstance(command, list) and command[:4] == [
        "renderer-python", "-m", "gs_video.scene.worker", "--request",
    ]
    assert "token" not in " ".join(command).lower()
    assert isinstance(options, dict) and options["shell"] is False
    assert result.frame_paths == (request.output_dir / "000001.png",)
    assert result.source_frame_indices == (0,)
    assert result.implementation_version == "gsplat-1.5.3"
    assert events == [(1, 1, "one")]
    assert not list(tmp_path.glob(".gs-video-renderer-*"))


def test_client_rejects_progress_after_terminal_duplicate_terminal_and_long_line(
    tmp_path: Path,
) -> None:
    bad_streams = [
        [
            '{"type":"complete","implementation_version":"v","outputs":[]}\n',
            '{"type":"progress","current":1,"total":1,"message":"late"}\n',
        ],
        [
            '{"type":"complete","implementation_version":"v","outputs":[]}\n',
            '{"type":"error","code":"system_error","message":"again"}\n',
        ],
        ["x" * (64 * 1024 + 1) + "\n"],
    ]
    for index, lines in enumerate(bad_streams):
        case = tmp_path / str(index)
        case.mkdir()
        request = _sequence_request(case)
        with pytest.raises(GsVideoError):
            _client(case, lambda *_args, **_kwargs: FakeProcess(lines)).render_sequence(
                request, lambda *_: None, CancellationToken()
            )


def test_cancel_terminates_renderer_process_tree(tmp_path: Path) -> None:
    request = _sequence_request(tmp_path)
    process = BlockingProcess()
    guard = RecordingGuard(process)
    token = CancellationToken()
    timer = threading.Timer(0.05, token.cancel)
    timer.start()
    try:
        with pytest.raises(CancelledError):
            _client(tmp_path, lambda *_args, **_kwargs: process, guard_factory=lambda _: guard).render_sequence(
                request, lambda *_: None, token
            )
    finally:
        timer.join()
    assert guard.calls and guard.calls[0] is False


def test_startup_gate_is_released_only_after_tree_guard_assignment(tmp_path: Path) -> None:
    request = _sequence_request(tmp_path)
    process = FakeProcess([
        '{"type":"progress","current":1,"total":1,"message":"one"}\n',
        '{"type":"complete","implementation_version":"v","outputs":["000001.png"]}\n'
    ])
    gates: list[Path] = []

    def factory(command: list[str], **_options: object) -> FakeProcess:
        gate = Path(command[command.index("--startup-gate") + 1])
        assert gate.read_text(encoding="utf-8") == "WAIT\n"
        gates.append(gate)
        request.output_dir.mkdir()
        Image.new("RGB", (64, 36)).save(request.output_dir / "000001.png")
        return process

    def guard_factory(actual: FakeProcess) -> RecordingGuard:
        assert actual is process
        assert gates[0].read_text(encoding="utf-8") == "WAIT\n"
        return RecordingGuard(process)

    _client(tmp_path, factory, guard_factory=guard_factory).render_sequence(
        request, lambda *_: None, CancellationToken()
    )
    assert not gates[0].exists()


def test_worker_is_registered_before_startup_gate_release(tmp_path: Path) -> None:
    request = _sequence_request(tmp_path)
    process = FakeProcess([
        '{"type":"progress","current":1,"total":1,"message":"one"}\n',
        '{"type":"complete","implementation_version":"v","outputs":["000001.png"]}\n',
    ])

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        request.output_dir.mkdir()
        Image.new("RGB", (64, 36)).save(request.output_dir / "000001.png")
        return process

    client = _client(tmp_path, factory)
    original_release = client._release_gate

    def release(gate: Path) -> None:
        assert len(client._active) == 1
        original_release(gate)

    client._release_gate = release  # type: ignore[method-assign]
    client.render_sequence(request, lambda *_: None, CancellationToken())


def test_renderer_client_rejects_wsl_prefix_until_json_paths_are_translated() -> None:
    with pytest.raises(ValueError, match="WSL"):
        RendererWorkerClient(worker_prefix=("wsl.exe", "--", "python"))


def test_sequence_rejects_output_path_escape_and_wrong_dimensions(tmp_path: Path) -> None:
    request = _sequence_request(tmp_path)
    outside = tmp_path / "outside.png"

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        request.output_dir.mkdir()
        Image.new("RGB", (32, 18)).save(request.output_dir / "000001.png")
        outside.write_bytes(b"outside")
        return FakeProcess([
            json.dumps({
                "type": "complete", "implementation_version": "v",
                "outputs": [str(outside.absolute())],
            }) + "\n"
        ])

    with pytest.raises(GsVideoError):
        _client(tmp_path, factory).render_sequence(request, lambda *_: None, CancellationToken())


def test_sequence_rejects_output_directory_replacement_during_validation(
    tmp_path: Path,
) -> None:
    request = _sequence_request(tmp_path)
    second_pose = OrbitCamera(
        target=(0.0, 0.0, 1.0), distance=1.0, yaw=5.0,
        pitch=0.0, fov_y_degrees=60.0,
    ).camera_to_world()
    first_pose = read_mapped_trajectory(request.camera_manifest).camera_to_world[0]
    write_mapped_trajectory(
        request.camera_manifest,
        MappedTrajectory(fov_y_degrees=60.0, camera_to_world=(first_pose, second_pose)),
    )
    process = FakeProcess([
        '{"type":"progress","current":1,"total":2,"message":"one"}\n',
        '{"type":"progress","current":2,"total":2,"message":"two"}\n',
        '{"type":"complete","implementation_version":"v","outputs":["000001.png","000002.png"]}\n',
    ])

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        request.output_dir.mkdir()
        Image.new("RGB", (64, 36)).save(request.output_dir / "000001.png")
        Image.new("RGB", (64, 36)).save(request.output_dir / "000002.png")
        return process

    client = _client(tmp_path, factory)
    original_validate = client._validate_png
    calls = 0

    def replacing_validate(path: Path, width: int, height: int) -> None:
        nonlocal calls
        original_validate(path, width, height)
        calls += 1
        if calls == 1:
            old = request.output_dir.with_name("old-frames")
            request.output_dir.replace(old)
            request.output_dir.mkdir()
            Image.new("RGB", (64, 36)).save(request.output_dir / "000001.png")
            Image.new("RGB", (64, 36)).save(request.output_dir / "000002.png")

    client._validate_png = replacing_validate  # type: ignore[method-assign]
    with pytest.raises(GsVideoError, match="directory identity"):
        client.render_sequence(request, lambda *_: None, CancellationToken())


@pytest.mark.parametrize("bad_depth", [np.nan, np.inf, -1.0])
def test_pick_rejects_nonfinite_or_negative_depth(tmp_path: Path, bad_depth: float) -> None:
    scene, _manifest = _inputs(tmp_path)
    output = (tmp_path / "pick.npz").absolute()
    request = RenderPickRequest(
        type="render_pick", scene_path=scene, output_npz=output,
        camera=OrbitCameraPayload(
            target=(0.0, 0.0, 1.0), distance=1.0, yaw=0.0,
            pitch=0.0, fov_y_degrees=60.0,
        ),
        width=64, height=36,
    )

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        np.savez(
            output,
            rgb=np.zeros((36, 64, 3), dtype=np.uint8),
            expected_depth=np.full((36, 64), bad_depth, dtype=np.float32),
        )
        return FakeProcess([
            json.dumps({
                "type": "complete", "implementation_version": "v",
                "outputs": [output.name],
            }) + "\n"
        ])

    with pytest.raises(GsVideoError):
        _client(tmp_path, factory).render_pick(request, CancellationToken())


def test_probe_returns_strict_identity(tmp_path: Path) -> None:
    process = FakeProcess([
        '{"type":"probe","torch":"2.7.1","gsplat":"1.5.3","device":"cuda"}\n'
    ])
    identity = _client(tmp_path, lambda *_args, **_kwargs: process).probe(
        ProbeRequest(type="probe"), CancellationToken()
    )
    assert identity.torch == "2.7.1"
    assert identity.gsplat == "1.5.3"
    assert identity.device == "cuda"


def test_stderr_log_is_bounded_to_one_mib(tmp_path: Path) -> None:
    process = FakeProcess([
        '{"type":"error","code":"system_error","message":"failed"}\n'
    ], stderr="x" * (2 * 1024 * 1024), returncode=1)
    with pytest.raises(GsVideoError, match="failed"):
        _client(tmp_path, lambda *_args, **_kwargs: process).probe(
            ProbeRequest(type="probe"), CancellationToken()
        )
    log = (tmp_path / "logs" / "renderer-worker.log").read_bytes()
    assert len(log) <= 1024 * 1024
    assert b"truncated" in log.lower()


def test_diagnostic_log_replaces_hardlink_without_modifying_its_target(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"keep")
    log = tmp_path / "logs" / "renderer-worker.log"
    log.parent.mkdir()
    os.link(victim, log)
    process = FakeProcess([
        '{"type":"probe","torch":"2","gsplat":"1","device":"cuda"}\n'
    ], stderr="diagnostic")

    identity = _client(tmp_path, lambda *_args, **_kwargs: process).probe()

    assert identity.device == "cuda"
    assert victim.read_bytes() == b"keep"
    assert log.read_bytes() == b"diagnostic"
    assert log.stat().st_ino != victim.stat().st_ino


def test_pick_rejects_oversized_npz_member_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, _manifest = _inputs(tmp_path)
    output = (tmp_path / "pick.npz").absolute()
    request = RenderPickRequest(
        type="render_pick", scene_path=scene, output_npz=output,
        camera=OrbitCameraPayload(
            target=(0.0, 0.0, 1.0), distance=1.0, yaw=0.0,
            pitch=0.0, fov_y_degrees=60.0,
        ),
        width=4, height=4,
    )

    def factory(*_args: object, **_kwargs: object) -> FakeProcess:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("rgb.npy", b"x" * (1024 * 1024))
            archive.writestr("expected_depth.npy", b"x")
        return FakeProcess([
            json.dumps({
                "type": "complete", "implementation_version": "v",
                "outputs": [output.name],
            }) + "\n"
        ])

    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not inspect oversized members")

    monkeypatch.setattr(worker_client_module.np, "load", forbidden_load)
    with pytest.raises(GsVideoError, match="member"):
        _client(tmp_path, factory).render_pick(request, CancellationToken())
    assert called is False


def test_shutdown_registry_terminates_a_live_renderer_tree(tmp_path: Path) -> None:
    request = _sequence_request(tmp_path)
    process = BlockingProcess()
    guard = RecordingGuard(process)
    client = _client(
        tmp_path,
        lambda *_args, **_kwargs: process,
        guard_factory=lambda _process: guard,
    )
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            client.render_sequence(request, lambda *_: None, CancellationToken())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    deadline = threading.Event()
    for _ in range(100):
        if client._active:
            break
        deadline.wait(0.01)
    asyncio.run(client.terminate_all())
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert guard.calls and guard.calls[0] is False
    assert errors and isinstance(errors[0], GsVideoError)
