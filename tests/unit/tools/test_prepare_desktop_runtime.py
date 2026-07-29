from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.prepare_desktop_runtime import (
    DesktopRuntimeError,
    prepare_desktop_runtime,
)


def create_runtime_layout(root: Path) -> None:
    required = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".runtime" / "segmentation" / ".venv" / "Scripts" / "python.exe",
        root / ".runtime" / "renderer" / ".venv" / "Scripts" / "python.exe",
        root
        / ".runtime"
        / "segmentation"
        / "EdgeTAM"
        / "sam2"
        / "configs"
        / "edgetam.yaml",
        root
        / ".runtime"
        / "segmentation"
        / "EdgeTAM"
        / "checkpoints"
        / "edgetam.pt",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime")


def test_prepare_desktop_runtime_writes_confined_absolute_configuration(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").absolute()
    root.mkdir()
    create_runtime_layout(root)

    runtime_path = prepare_desktop_runtime(root, validate=False)

    assert runtime_path == root / ".runtime" / "desktop-runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_root = root / ".runtime"
    assert payload == {
        "project_root": str(runtime_root / "projects" / "default"),
        "model_root": str(runtime_root / "segmentation" / "EdgeTAM"),
        "segmentation_backend": "edgetam",
        "segmentation_worker_prefix": [
            str(runtime_root / "segmentation" / ".venv" / "Scripts" / "python.exe")
        ],
        "segmentation_model_config": str(
            runtime_root
            / "segmentation"
            / "EdgeTAM"
            / "sam2"
            / "configs"
            / "edgetam.yaml"
        ),
        "segmentation_checkpoint": str(
            runtime_root
            / "segmentation"
            / "EdgeTAM"
            / "checkpoints"
            / "edgetam.pt"
        ),
        "renderer_worker_prefix": [
            str(runtime_root / "renderer" / ".venv" / "Scripts" / "python.exe")
        ],
        "renderer_sh_degree": 3,
        "available_vram_limit_mb": 8192,
    }
    assert (runtime_root / "projects" / "default").is_dir()


def test_prepare_desktop_runtime_does_not_rewrite_unchanged_file(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").absolute()
    root.mkdir()
    create_runtime_layout(root)
    runtime_path = prepare_desktop_runtime(root, validate=False)
    original_mtime = runtime_path.stat().st_mtime_ns

    assert prepare_desktop_runtime(root, validate=False) == runtime_path
    assert runtime_path.stat().st_mtime_ns == original_mtime


def test_prepare_desktop_runtime_reports_missing_project_python(tmp_path: Path) -> None:
    root = (tmp_path / "repo").absolute()
    root.mkdir()

    with pytest.raises(DesktopRuntimeError, match=r"\.venv.*python\.exe"):
        prepare_desktop_runtime(root, validate=False)
