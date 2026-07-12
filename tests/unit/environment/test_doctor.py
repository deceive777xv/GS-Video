import builtins
import sys
from types import SimpleNamespace

import pytest

from gs_video.environment.doctor import EnvironmentDoctor, probe_cuda


def test_doctor_reports_missing_commands_without_starting_gpu() -> None:
    doctor = EnvironmentDoctor(which=lambda name: None, cuda_probe=lambda: (False, 0))
    report = doctor.check()
    assert report.ready is False
    assert {issue.code for issue in report.issues} == {
        "ffmpeg_missing",
        "ffprobe_missing",
        "cuda_unavailable",
    }


def test_doctor_reports_ready_environment() -> None:
    doctor = EnvironmentDoctor(which=lambda name: f"C:/{name}.exe", cuda_probe=lambda: (True, 8192))

    report = doctor.check()

    assert report.ready is True
    assert report.vram_mb == 8192
    assert report.issues == []


def test_doctor_preserves_vram_when_only_ffprobe_is_missing() -> None:
    doctor = EnvironmentDoctor(
        which=lambda name: "C:/ffmpeg.exe" if name == "ffmpeg" else None,
        cuda_probe=lambda: (True, 6144),
    )

    report = doctor.check()

    assert report.ready is False
    assert report.vram_mb == 6144
    assert [issue.code for issue in report.issues] == ["ffprobe_missing"]


def test_probe_cuda_reports_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def import_without_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_torch)

    assert probe_cuda() == (False, 0)


def test_probe_cuda_reports_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda index: pytest.fail("GPU properties must not be queried"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert probe_cuda() == (False, 0)


def test_probe_cuda_converts_total_vram_to_mib(monkeypatch: pytest.MonkeyPatch) -> None:
    total_memory = 8 * 1024 * 1024 * 1024
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(total_memory=total_memory),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert probe_cuda() == (True, 8192)
