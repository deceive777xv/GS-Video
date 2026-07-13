from __future__ import annotations

import io
import subprocess
import threading
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
        (output / name).write_bytes(b"mask")
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
    (output / "000001.png").write_bytes(b"mask")
    calls: list[list[str]] = []
    client = _client(tmp_path, process)
    client.worker_prefix = ("wsl.exe", "-d", "Ubuntu", "--", "/opt/edgetam/bin/python")
    client._process_factory = lambda command, **_: (calls.append(command), process)[1]
    frames = _frames(tmp_path, ("000001.jpg",))
    client.segment(frames, Prompt(0, 1, 1), output, lambda *_: None, CancellationToken())
    command = calls[0]
    assert command[:7] == [
        "wsl.exe", "-d", "Ubuntu", "--", "/opt/edgetam/bin/python", "-m",
        "gs_video.segmentation.worker",
    ]
    assert command[command.index("--frames") + 1].startswith("/mnt/c/")
    assert command[command.index("--output") + 1].startswith("/mnt/c/")
    assert command[command.index("--config") + 1].startswith("/mnt/c/")
    assert command[command.index("--checkpoint") + 1].startswith("/mnt/c/")
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
    (output / "000001.png").write_bytes(b"mask")
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
    (output / "000001.png").write_bytes(b"mask")
    with pytest.raises(GsVideoError):
        _client(tmp_path, FakeProcess(lines, returncode=returncode)).segment(
            _frames(tmp_path, ("000001.jpg",)), Prompt(0, 1, 1), output,
            lambda *_: None, CancellationToken(),
        )
