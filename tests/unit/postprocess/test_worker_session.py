from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

from gs_video.postprocess import worker


class _BinaryInput:
    def __init__(self, payload: bytes) -> None:
        self.buffer = BytesIO(payload)


def test_warm_worker_session_processes_multiple_requests(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    processed: list[Path] = []
    commands = b'{"request":"C:/preview/one.json"}\n{"request":"C:/preview/two.json"}\n'
    output = StringIO()
    monkeypatch.setattr(worker, "_configure_cuda", lambda: None)
    monkeypatch.setattr(worker, "_process", processed.append)
    monkeypatch.setattr(worker.sys, "stdin", _BinaryInput(commands))
    monkeypatch.setattr(worker.sys, "stdout", output)

    assert worker.run_session() == 0
    assert processed == [Path("C:/preview/one.json"), Path("C:/preview/two.json")]
    assert '"type": "ready"' in output.getvalue()


def test_warm_worker_session_rejects_unbounded_command_shape(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    output = StringIO()
    monkeypatch.setattr(worker, "_configure_cuda", lambda: None)
    monkeypatch.setattr(worker.sys, "stdin", _BinaryInput(b'{"other":"value"}\n'))
    monkeypatch.setattr(worker.sys, "stdout", output)

    assert worker.run_session() == 1
    assert '"type": "error"' in output.getvalue()
