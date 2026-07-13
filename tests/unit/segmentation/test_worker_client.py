from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from gs_video.domain.contracts import MaskSequence, Prompt, SegmentationBackend
from gs_video.domain.errors import CancelledError, GsVideoError, UnsupportedMaterialError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.segmentation.client import VideoSegmenterClient
from gs_video.segmentation.paths import worker_path


class FakeProcess:
    def __init__(self, lines: list[str], *, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = io.StringIO("".join(lines))
        self.stderr = io.StringIO(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = self._final_returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _frames(tmp_path: Path, names: tuple[str, ...] = ("000002.jpg", "000010.png")) -> list[Path]:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    result = []
    for name in names:
        path = frames_dir / name
        Image.new("RGB", (32, 32)).save(path)
        result.append(path)
    return result


def _write_mask(path: Path, *, mode: str = "L", size: tuple[int, int] = (32, 32)) -> None:
    Image.new(mode, size).save(path, format="PNG")


def _client(tmp_path: Path, process: FakeProcess, **kwargs: Any) -> VideoSegmenterClient:
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_text("model: fake", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    return VideoSegmenterClient(
        backend=SegmentationBackend.EDGETAM,
        worker_prefix=("python",),
        model_config=config,
        checkpoint=checkpoint,
        process_factory=lambda *args, **options: process,
        **kwargs,
    )


def test_client_constructs_native_command_without_shell_and_parses_result(tmp_path: Path) -> None:
    output = tmp_path / "masks"
    output.mkdir()
    for name in ("000002.png", "000010.png"):
        _write_mask(output / name)
    process = FakeProcess(
        [
            '{"type":"progress","current":1,"total":2}\n',
            '{"type":"progress","current":2,"total":2}\n',
            '{"type":"result","mask_dir":"masks","frames":2}\n',
        ]
    )
    calls: list[tuple[list[str], dict[str, object]]] = []
    client = _client(tmp_path, process)
    client._process_factory = lambda command, **options: (calls.append((command, options)), process)[1]
    progress: list[tuple[int, int, str]] = []

    result = client.segment(
        list(reversed(_frames(tmp_path))),
        Prompt(frame_index=0, x=10, y=20),
        output,
        lambda *event: progress.append(event),
        CancellationToken(),
    )

    assert result == MaskSequence(mask_dir=output, frame_count=2)
    command, options = calls[0]
    assert command[:4] == ["python", "-m", "gs_video.segmentation.worker", "--backend"]
    assert command[4:6] == ["edgetam", "--frames"]
    assert command[command.index("--point") + 1] == "10,20"
    assert options["shell"] is False
    assert progress == [(1, 2, "分割前景 1/2"), (2, 2, "分割前景 2/2")]


def test_client_preserves_wsl_prefix(tmp_path: Path) -> None:
    process = FakeProcess([
        '{"type":"progress","current":1,"total":1}\n',
        '{"type":"result","mask_dir":"masks","frames":1}\n',
    ])
    output = tmp_path / "masks"
    output.mkdir()
    _write_mask(output / "000001.png")
    calls: list[list[str]] = []
    client = _client(tmp_path, process)
    prefix = ("wsl.exe", "-d", "Ubuntu", "--", "/opt/edgetam/bin/python")
    client.worker_prefix = prefix
    client._process_factory = lambda command, **_: (calls.append(command), process)[1]
    frames = _frames(tmp_path, ("000001.jpg",))
    client.segment(frames, Prompt(0, 1, 1), output, lambda *_: None, CancellationToken())
    command = calls[0]
    assert command[:7] == [
        "wsl.exe", "-d", "Ubuntu", "--", "/opt/edgetam/bin/python", "-m",
        "gs_video.segmentation.worker",
    ]
    expected_paths = {
        "--frames": worker_path(frames[0].parent, prefix),
        "--output": worker_path(output, prefix),
        "--config": worker_path(client.model_config, prefix),
        "--checkpoint": worker_path(client.checkpoint, prefix),
    }
    for option, expected in expected_paths.items():
        assert command[command.index(option) + 1] == expected
    assert "\\" not in " ".join(command)


def test_wsl_path_translation_rejects_relative_and_unc_paths() -> None:
    prefix = ("wsl.exe", "--", "/opt/venv/bin/python")
    assert worker_path(Path(r"E:\Project\GS-Video\model.pt"), prefix) == (
        "/mnt/e/Project/GS-Video/model.pt"
    )
    for path in (Path("relative/model.pt"), Path(r"\\server\share\model.pt")):
        with pytest.raises(ValueError, match="WSL"):
            worker_path(path, prefix)


@pytest.mark.parametrize(
    "lines",
    [
        ["not json\n"],
        ['{"type":"mystery"}\n'],
        ['{"type":"progress","current":2,"total":2}\n', '{"type":"progress","current":1,"total":2}\n'],
        ['{"type":"result","mask_dir":"masks","frames":0}\n', '{"type":"result","mask_dir":"masks","frames":0}\n'],
        ['{"type":"progress","current":1,"total":2,"extra":true}\n'],
        ['{"type":"progress","current":true,"total":1}\n', '{"type":"result","mask_dir":"masks","frames":0}\n'],
    ],
)
def test_client_rejects_malformed_protocol(tmp_path: Path, lines: list[str]) -> None:
    output = tmp_path / "masks"
    output.mkdir()
    _write_mask(output / "000001.png")
    with pytest.raises(GsVideoError, match="worker"):
        _client(tmp_path, FakeProcess(lines)).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), output,
            lambda *_: None, CancellationToken()
        )


def test_client_maps_worker_error_and_logs_stderr(tmp_path: Path) -> None:
    process = FakeProcess(
        ['{"type":"error","code":"unsupported_material","message":"主要人物长时间不可见"}\n'],
        stderr="library chatter\n",
    )
    with pytest.raises(UnsupportedMaterialError, match="主要人物长时间不可见"):
        _client(tmp_path, process).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, CancellationToken()
        )
    assert "library chatter" in (tmp_path / "logs" / "segmentation-worker.log").read_text(
        encoding="utf-8"
    )


def test_client_rejects_nonzero_exit_path_traversal_and_missing_masks(tmp_path: Path) -> None:
    for process, output, message in (
        (FakeProcess([], returncode=3), tmp_path / "one", "退出"),
        (FakeProcess([
            '{"type":"progress","current":1,"total":1}\n',
            '{"type":"result","mask_dir":"../escape","frames":1}\n',
        ]), tmp_path / "two", "路径"),
        (FakeProcess([
            '{"type":"progress","current":1,"total":1}\n',
            '{"type":"result","mask_dir":"three","frames":1}\n',
        ]), tmp_path / "three", "mask"),
    ):
        output.mkdir()
        with pytest.raises(GsVideoError, match=message):
            _client(tmp_path, process).segment(
                _frames(tmp_path, ("000002.jpg",)),
                Prompt(0, 1, 1), output, lambda *_: None, CancellationToken()
            )


def test_client_cancellation_terminates_and_reaps(tmp_path: Path) -> None:
    process = FakeProcess([])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        _client(tmp_path, process).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, token
        )
    assert process.terminated
    assert process.returncode is not None


def test_client_cancellation_kills_after_timeout(tmp_path: Path) -> None:
    class StubbornProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            return super().wait(timeout)

        def terminate(self) -> None:
            self.terminated = True

    process = StubbornProcess([])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        _client(tmp_path, process).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, token
        )
    assert process.terminated and process.killed and process.returncode is not None


def test_client_keeps_polling_cancellation_after_stdout_eof(tmp_path: Path) -> None:
    class NeverExitProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            return super().wait(timeout)

        def terminate(self) -> None:
            self.terminated = True

    process = NeverExitProcess([])
    token = CancellationToken()
    timer = threading.Timer(0.05, token.cancel)
    timer.start()
    try:
        with pytest.raises(CancelledError):
            _client(tmp_path, process).segment(
                _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
                lambda *_: None, token
            )
    finally:
        timer.cancel()
    assert process.terminated and process.killed


def test_client_cancels_worker_that_hangs_after_terminal_event(tmp_path: Path) -> None:
    class TerminalThenHang(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            return super().wait(timeout)

        def terminate(self) -> None:
            self.terminated = True

    process = TerminalThenHang(
        ['{"type":"error","code":"unsupported_material","message":"x"}\n']
    )
    token = CancellationToken()
    timer = threading.Timer(0.05, token.cancel)
    timer.start()
    try:
        with pytest.raises(CancelledError):
            _client(tmp_path, process).segment(
                _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
                lambda *_: None, token,
            )
    finally:
        timer.cancel()
    assert process.terminated and process.killed


def test_client_rejects_empty_partial_inventory_and_invalid_prompt(tmp_path: Path) -> None:
    process = FakeProcess([])
    client = _client(tmp_path, process)
    with pytest.raises(GsVideoError, match="至少一帧"):
        client.segment([], Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None, CancellationToken())
    frames = _frames(tmp_path, ("000001.jpg", "000002.png"))
    with pytest.raises(GsVideoError, match="完整目录"):
        client.segment(
            frames[:1], Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None,
            CancellationToken(),
        )
    for prompt in (Prompt(True, 1, 1), Prompt(2, 1, 1), Prompt(0, -1, 1), Prompt(0, 32, 1)):
        with pytest.raises(GsVideoError, match="提示"):
            client.segment(
                frames, prompt, tmp_path / "masks", lambda *_: None, CancellationToken()
            )


def test_client_rejects_nonnumeric_image_in_worker_directory(tmp_path: Path) -> None:
    frames = _frames(tmp_path, ("000001.jpg",))
    Image.new("RGB", (32, 32)).save(frames[0].parent / "poster.jpg")
    client = _client(tmp_path, FakeProcess([]))
    client._process_factory = lambda *args, **kwargs: pytest.fail("worker must not start")
    with pytest.raises(GsVideoError, match="数字"):
        client.segment(
            frames, Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None,
            CancellationToken(),
        )


def test_client_rejects_output_overlap_with_frames_or_model(tmp_path: Path) -> None:
    frames = _frames(tmp_path, ("000001.jpg",))
    client = _client(tmp_path, FakeProcess([]))
    for output in (frames[0].parent / "masks", client.model_config, client.checkpoint):
        with pytest.raises(GsVideoError, match="输出"):
            client.segment(frames, Prompt(0, 1, 1), output, lambda *_: None, CancellationToken())


def test_client_missing_pipe_is_cleaned_up(tmp_path: Path) -> None:
    process = FakeProcess([])
    process.stdout = None
    with pytest.raises(GsVideoError, match="管道"):
        _client(tmp_path, process).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, CancellationToken(),
        )
    assert process.terminated and process.returncode is not None


def test_client_drains_large_stderr_before_return(tmp_path: Path) -> None:
    stderr = "x" * (2 * 1024 * 1024)
    process = FakeProcess(
        ['{"type":"error","code":"system_error","message":"模型崩溃"}\n'],
        stderr=stderr,
        returncode=1,
    )
    with pytest.raises(GsVideoError, match="模型崩溃"):
        _client(tmp_path, process).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, CancellationToken(),
        )
    assert (tmp_path / "logs" / "segmentation-worker.log").stat().st_size == len(stderr)


@pytest.mark.parametrize(
    ("lines", "returncode"),
    [
        (['{"type":"error","code":"unsupported_material","message":"x"}\n'], 1),
        (['{"type":"error","code":"system_error","message":"x"}\n'], 0),
        (['{"type":"error","code":"unknown","message":"x"}\n'], 1),
        (['{"type":"result","mask_dir":"masks","frames":1}\n'], 0),
        ([
            '{"type":"progress","current":1,"total":2}\n',
            '{"type":"result","mask_dir":"masks","frames":1}\n',
        ], 0),
    ],
)
def test_client_rejects_terminal_exit_or_progress_mismatch(
    tmp_path: Path, lines: list[str], returncode: int
) -> None:
    output = tmp_path / "masks"
    output.mkdir()
    _write_mask(output / "000001.png")
    with pytest.raises(GsVideoError):
        _client(tmp_path, FakeProcess(lines, returncode=returncode)).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), output,
            lambda *_: None, CancellationToken(),
        )


def test_client_rejects_corrupt_rgb_or_wrong_size_masks(tmp_path: Path) -> None:
    frames = _frames(tmp_path, ("000001.jpg",))
    variants: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("corrupt", lambda path: path.write_bytes(b"not-png")),
        ("rgb", lambda path: _write_mask(path, mode="RGB")),
        ("wrong-size", lambda path: _write_mask(path, size=(16, 16))),
    )
    for name, write in variants:
        output = tmp_path / name
        output.mkdir()
        write(output / "000001.png")
        process = FakeProcess([
            '{"type":"progress","current":1,"total":1}\n',
            f'{{"type":"result","mask_dir":"{name}","frames":1}}\n',
        ])
        with pytest.raises(GsVideoError, match="mask"):
            _client(tmp_path, process).segment(
                frames, Prompt(0, 1, 1), output, lambda *_: None, CancellationToken()
            )


def test_client_rejects_symlinked_frame_ancestor_when_supported(tmp_path: Path) -> None:
    real_frames = tmp_path / "real-frames"
    real_frames.mkdir()
    Image.new("RGB", (32, 32)).save(real_frames / "000001.jpg")
    linked_frames = tmp_path / "linked-frames"
    try:
        linked_frames.symlink_to(real_frames, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not available")
    with pytest.raises(GsVideoError, match="链接"):
        _client(tmp_path, FakeProcess([])).segment(
            [linked_frames / "000001.jpg"], Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, CancellationToken(),
        )


def test_client_cancellation_does_not_deadlock_on_inherited_pipe_handles(tmp_path: Path) -> None:
    class HangingStream:
        def __init__(self) -> None:
            self.released = threading.Event()

        def __iter__(self) -> "HangingStream":
            return self

        def __next__(self) -> str:
            self.released.wait()
            raise StopIteration

        def read(self) -> str:
            self.released.wait()
            return ""

        def close(self) -> None:
            self.released.set()

    process = FakeProcess([])
    process.stdout = HangingStream()  # type: ignore[assignment]
    process.stderr = HangingStream()  # type: ignore[assignment]
    token = CancellationToken()
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            _client(tmp_path, process).segment(
                _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
                lambda *_: None, token,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    token.cancel()
    thread.join(timeout=3)
    stuck = thread.is_alive()
    process.stdout.close()
    process.stderr.close()
    thread.join(timeout=1)

    assert not stuck
    assert errors and isinstance(errors[0], CancelledError)
    assert "截断" in (tmp_path / "logs" / "segmentation-worker.log").read_text(encoding="utf-8")


def test_client_terminates_process_tree_before_parent_exits(tmp_path: Path) -> None:
    class RecordingGuard:
        def terminate(self, *, force: bool) -> bool:
            tree_calls.append(force)
            return True

        def close(self) -> None:
            return

    process = FakeProcess([])
    tree_calls: list[bool] = []
    client = _client(tmp_path, process, tree_guard_factory=lambda _: RecordingGuard())
    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        client.segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, token,
        )

    assert tree_calls == [False]


def test_client_reaps_worker_when_tree_guard_assignment_fails(tmp_path: Path) -> None:
    process = FakeProcess([])

    def fail_guard(actual_process: object) -> object:
        del actual_process
        raise OSError("job assignment failed")

    client = _client(tmp_path, process, tree_guard_factory=fail_guard)
    with pytest.raises(GsVideoError, match="进程树隔离"):
        client.segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), tmp_path / "masks",
            lambda *_: None, CancellationToken(),
        )
    assert process.terminated and process.returncode is not None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_real_job_guard_kills_child_after_parent_exits(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "stdout=sys.stdout,stderr=sys.stderr);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(0.5)"
    )

    def factory(command: list[str], **options: object) -> subprocess.Popen[str]:
        del command
        return subprocess.Popen(  # type: ignore[call-overload]
            [sys.executable, "-c", parent_code, str(child_pid_path)], **options
        )

    frames = _frames(tmp_path, ("000001.jpg",))
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_text("model", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    client = VideoSegmenterClient(
        backend=SegmentationBackend.EDGETAM,
        worker_prefix=(sys.executable,), model_config=config, checkpoint=checkpoint,
        process_factory=factory,
    )
    token = CancellationToken()
    timer = threading.Timer(1.0, token.cancel)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(CancelledError):
            client.segment(frames, Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None, token)
    finally:
        timer.cancel()

    assert time.monotonic() - started < 5
    child_pid = int(child_pid_path.read_text())
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, child_pid)
    if handle:
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        assert exit_code.value != 259
