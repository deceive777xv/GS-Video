import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gs_video.domain.contracts import SegmentationBackend
from gs_video.environment.doctor import EnvironmentDoctor, probe_cuda
from gs_video.segmentation.paths import worker_path


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


def test_doctor_checks_selected_segmentation_assets_and_probe(tmp_path: Path) -> None:
    config = tmp_path / "edgetam.yaml"
    checkpoint = tmp_path / "edgetam.pt"
    config.write_text("model", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    commands: list[list[str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0,
            json.dumps({
                "type": "probe", "backend": "edgetam",
                "config": str(config.resolve()), "checkpoint": str(checkpoint.resolve()),
                "predictor": "sam2.sam2_video_predictor.SAM2VideoPredictor",
            }) + "\n", "",
        )

    report = EnvironmentDoctor(
        which=lambda name: f"C:/{name}.exe",
        cuda_probe=lambda: (True, 8192),
        segmentation_backend=SegmentationBackend.EDGETAM,
        worker_prefix=("python",),
        model_config=config,
        checkpoint=checkpoint,
        process_runner=run,
    ).check()

    assert report.ready
    assert commands[0][:4] == ["python", "-m", "gs_video.segmentation.worker", "--probe"]


def test_doctor_rejects_probe_identity_mismatch_and_missing_asset(tmp_path: Path) -> None:
    config = tmp_path / "edgetam.yaml"
    config.write_text("model", encoding="utf-8")
    checkpoint = tmp_path / "missing.pt"
    report = EnvironmentDoctor(
        which=lambda name: f"C:/{name}.exe",
        cuda_probe=lambda: (True, 8192),
        segmentation_backend=SegmentationBackend.EDGETAM,
        worker_prefix=("python",), model_config=config, checkpoint=checkpoint,
        process_runner=lambda command, **options: subprocess.CompletedProcess(
            command, 0, '{"type":"probe","backend":"sam2","config":"x","checkpoint":"y"}\n', ""
        ),
    ).check()
    assert "segmentation_checkpoint_unreadable" in [issue.code for issue in report.issues]


def test_doctor_rejects_probe_identity_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "edgetam.yaml"
    checkpoint = tmp_path / "edgetam.pt"
    config.write_text("model", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    report = EnvironmentDoctor(
        which=lambda name: f"C:/{name}.exe",
        cuda_probe=lambda: (True, 8192),
        segmentation_backend=SegmentationBackend.EDGETAM,
        worker_prefix=("python",), model_config=config, checkpoint=checkpoint,
        process_runner=lambda command, **options: subprocess.CompletedProcess(
            command, 0,
            json.dumps({
                "type": "probe", "backend": "sam2", "config": str(config.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "predictor": "sam2.sam2_video_predictor.SAM2VideoPredictor",
            }) + "\n", "",
        ),
    ).check()
    assert "segmentation_probe_failed" in [issue.code for issue in report.issues]


def test_doctor_translates_wsl_probe_asset_arguments(tmp_path: Path) -> None:
    config = tmp_path / "edgetam.yaml"
    checkpoint = tmp_path / "edgetam.pt"
    config.write_text("model", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    commands: list[list[str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        config_arg = command[command.index("--config") + 1]
        checkpoint_arg = command[command.index("--checkpoint") + 1]
        return subprocess.CompletedProcess(
            command, 0,
            json.dumps({
                "type": "probe", "backend": "edgetam", "config": config_arg,
                "checkpoint": checkpoint_arg,
                "predictor": "sam2.sam2_video_predictor.SAM2VideoPredictor",
            }) + "\n", "",
        )

    prefix = ("wsl.exe", "-d", "Ubuntu", "--", "/opt/edgetam/bin/python")
    report = EnvironmentDoctor(
        which=lambda name: f"C:/{name}.exe",
        cuda_probe=lambda: (True, 8192),
        segmentation_backend=SegmentationBackend.EDGETAM,
        worker_prefix=prefix,
        model_config=config, checkpoint=checkpoint, process_runner=run,
    ).check()

    assert report.ready
    command = commands[0]
    assert command[command.index("--config") + 1] == worker_path(config, prefix)
    assert command[command.index("--checkpoint") + 1] == worker_path(checkpoint, prefix)
