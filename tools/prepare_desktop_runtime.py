from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


class DesktopRuntimeError(RuntimeError):
    """Raised when the project-local desktop runtime is incomplete."""


def _required_file(path: Path, label: str) -> Path:
    absolute = path.absolute()
    if not absolute.is_file() or absolute.is_symlink():
        raise DesktopRuntimeError(f"{label} is missing: {absolute}")
    return absolute


def _runtime_payload(repo_root: Path) -> dict[str, Any]:
    root = repo_root.absolute()
    runtime_root = root / ".runtime"
    edgetam_root = runtime_root / "segmentation" / "EdgeTAM"
    _required_file(root / ".venv" / "Scripts" / "python.exe", "project Python")
    segmentation_python = _required_file(
        runtime_root / "segmentation" / ".venv" / "Scripts" / "python.exe",
        "segmentation worker Python",
    )
    renderer_python = _required_file(
        runtime_root / "renderer" / ".venv" / "Scripts" / "python.exe",
        "renderer worker Python",
    )
    model_config = _required_file(
        edgetam_root / "sam2" / "configs" / "edgetam.yaml",
        "EdgeTAM model config",
    )
    checkpoint = _required_file(
        edgetam_root / "checkpoints" / "edgetam.pt",
        "EdgeTAM checkpoint",
    )
    return {
        "project_root": str(runtime_root / "projects" / "default"),
        "model_root": str(edgetam_root),
        "segmentation_backend": "edgetam",
        "segmentation_worker_prefix": [str(segmentation_python)],
        "segmentation_model_config": str(model_config),
        "segmentation_checkpoint": str(checkpoint),
        "renderer_worker_prefix": [str(renderer_python)],
        "renderer_sh_degree": 3,
        "available_vram_limit_mb": 8192,
    }


def prepare_desktop_runtime(repo_root: Path, *, validate: bool = True) -> Path:
    root = repo_root.absolute()
    if not root.is_dir():
        raise DesktopRuntimeError(f"repository root is unavailable: {root}")
    payload = _runtime_payload(root)
    runtime_root = root / ".runtime"
    project_root = runtime_root / "projects" / "default"
    project_root.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_root / "desktop-runtime.json"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    current = None
    try:
        current = runtime_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    if current != serialized:
        temporary = runtime_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            os.replace(temporary, runtime_path)
        finally:
            temporary.unlink(missing_ok=True)
    if validate:
        source_root = root / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from gs_video.runtime import load_runtime_config

        try:
            load_runtime_config(runtime_path)
        except (OSError, ValueError) as error:
            raise DesktopRuntimeError(f"desktop runtime is invalid: {error}") from error
    return runtime_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = prepare_desktop_runtime(args.repo_root)
    except DesktopRuntimeError as error:
        raise SystemExit(str(error)) from error
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
