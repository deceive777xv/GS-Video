from __future__ import annotations

import io
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from gs_video.domain.contracts import MaskSequence, Prompt, SegmentationBackend
from gs_video.domain.errors import CancelledError, GsVideoError, UnsupportedMaterialError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.segmentation.client import VideoSegmenterClient


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
    result = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"frame")
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
    process = FakeProcess(['{"type":"result","mask_dir":"masks","frames":0}\n'])
    output = tmp_path / "masks"
    output.mkdir()
    calls: list[list[str]] = []
    client = _client(tmp_path, process)
    client.worker_prefix = ("wsl.exe", "-d", "Ubuntu", "--", "/opt/edgetam/bin/python")
    client._process_factory = lambda command, **_: (calls.append(command), process)[1]
    client.segment([], Prompt(0, 1, 1), output, lambda *_: None, CancellationToken())
    assert calls[0][:7] == [
        "wsl.exe", "-d", "Ubuntu", "--", "/opt/edgetam/bin/python", "-m",
        "gs_video.segmentation.worker",
    ]


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
    with pytest.raises(GsVideoError, match="worker"):
        _client(tmp_path, FakeProcess(lines)).segment(
            [], Prompt(0, 1, 1), output, lambda *_: None, CancellationToken()
        )


def test_client_maps_worker_error_and_logs_stderr(tmp_path: Path) -> None:
    process = FakeProcess(
        ['{"type":"error","code":"unsupported_material","message":"主要人物长时间不可见"}\n'],
        stderr="library chatter\n",
    )
    with pytest.raises(UnsupportedMaterialError, match="主要人物长时间不可见"):
        _client(tmp_path, process).segment(
            [], Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None, CancellationToken()
        )
    assert "library chatter" in (tmp_path / "logs" / "segmentation-worker.log").read_text(
        encoding="utf-8"
    )


def test_client_rejects_nonzero_exit_path_traversal_and_missing_masks(tmp_path: Path) -> None:
    for process, output, message in (
        (FakeProcess([], returncode=3), tmp_path / "one", "退出"),
        (FakeProcess(['{"type":"result","mask_dir":"../escape","frames":0}\n']), tmp_path / "two", "路径"),
        (FakeProcess(['{"type":"result","mask_dir":"three","frames":1}\n']), tmp_path / "three", "mask"),
    ):
        output.mkdir()
        with pytest.raises(GsVideoError, match=message):
            _client(tmp_path, process).segment(
                _frames(tmp_path, ("000002.jpg",)) if message == "mask" else [],
                Prompt(0, 1, 1), output, lambda *_: None, CancellationToken()
            )


def test_client_cancellation_terminates_and_reaps(tmp_path: Path) -> None:
    process = FakeProcess([])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        _client(tmp_path, process).segment(
            [], Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None, token
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
            [], Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None, token
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
                [], Prompt(0, 1, 1), tmp_path / "masks", lambda *_: None, token
            )
    finally:
        timer.cancel()
    assert process.terminated and process.killed
