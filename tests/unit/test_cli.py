import io
import json
from pathlib import Path

import pytest

from gs_video import __main__ as cli
from gs_video.environment.doctor import EnvironmentIssue, EnvironmentReport


class StubDoctor:
    def __init__(self, report: EnvironmentReport) -> None:
        self._report = report

    def check(self) -> EnvironmentReport:
        return self._report


def test_doctor_json_returns_zero_when_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = EnvironmentReport(ready=True, vram_mb=8192, issues=[])
    monkeypatch.setattr(cli, "EnvironmentDoctor", lambda: StubDoctor(report))

    exit_code = cli.main(["--doctor", "--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == report.model_dump(mode="json")


def test_doctor_json_returns_two_when_not_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = EnvironmentReport(
        ready=False,
        vram_mb=0,
        issues=[EnvironmentIssue(code="cuda_unavailable", message="CUDA is not available")],
    )
    monkeypatch.setattr(cli, "EnvironmentDoctor", lambda: StubDoctor(report))

    exit_code = cli.main(["--doctor", "--json"])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out) == report.model_dump(mode="json")


def test_no_option_prints_help_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([])

    assert exit_code == 0
    assert "usage:" in capsys.readouterr().out


def test_serve_refuses_missing_runtime_configuration(tmp_path: Path) -> None:
    missing = (tmp_path / "runtime.json").absolute()

    with pytest.raises(SystemExit, match="runtime configuration"):
        cli.main(["--serve", "--runtime-config", str(missing), "--session-token-stdin"])


def test_serve_loads_runtime_and_reads_one_bounded_private_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = (tmp_path / "runtime.json").absolute()
    runtime.write_text("{}", encoding="utf-8")
    loaded = object()
    calls: list[tuple[object, str]] = []

    def fake_run_api(config: object, token: str) -> int:
        calls.append((config, token))
        return 7

    monkeypatch.setattr(cli, "load_runtime_config", lambda path: loaded)
    monkeypatch.setattr("gs_video.app.run_api", fake_run_api)
    monkeypatch.setattr("sys.stdin", io.StringIO("private-token\nignored\n"))

    exit_code = cli.main(
        ["--serve", "--runtime-config", str(runtime), "--session-token-stdin"]
    )

    assert exit_code == 7
    assert calls == [(loaded, "private-token")]
    captured = capsys.readouterr()
    assert "private-token" not in captured.out
    assert "private-token" not in captured.err


def test_serve_requires_runtime_config_and_private_stdin_token() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--serve"])
