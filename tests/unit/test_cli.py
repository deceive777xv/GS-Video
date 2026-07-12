import json

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


def test_serve_uses_default_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_run_api(host: str, port: int) -> int:
        calls.append((host, port))
        return 7

    monkeypatch.setattr("gs_video.app.run_api", fake_run_api)

    exit_code = cli.main(["--serve"])

    assert exit_code == 7
    assert calls == [("127.0.0.1", 0)]
