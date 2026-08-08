from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


class DesktopRuntimeError(RuntimeError):
    """Raised when the project-local desktop runtime is incomplete."""


def _confined_path(
    path: Path,
    label: str,
    repo_root: Path,
    *,
    strict: bool = True,
) -> Path:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=strict)
    except OSError as error:
        raise DesktopRuntimeError(f"{label} is missing: {absolute}") from error
    if not resolved.is_relative_to(repo_root):
        raise DesktopRuntimeError(f"{label} resolves outside the repository: {absolute}")
    return resolved


def _required_file(
    path: Path,
    label: str,
    repo_root: Path,
    *,
    allow_missing: bool = False,
) -> Path:
    resolved = _confined_path(path, label, repo_root, strict=not allow_missing)
    if resolved.exists() and (not resolved.is_file() or path.is_symlink()):
        raise DesktopRuntimeError(f"{label} is missing: {path.absolute()}")
    if not resolved.exists() and not allow_missing:
        raise DesktopRuntimeError(f"{label} is missing: {path.absolute()}")
    return resolved


def _runtime_payload(
    repo_root: Path, *, allow_missing_resources: bool = False
) -> dict[str, Any]:
    try:
        root = repo_root.resolve(strict=True)
    except OSError as error:
        raise DesktopRuntimeError(
            f"repository root is unavailable: {repo_root.absolute()}"
        ) from error
    source_root = root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from gs_video.environment.vram import load_user_vram_limit

    _required_file(root / ".venv" / "Scripts" / "python.exe", "project Python", root)
    runtime_root = _confined_path(root / ".runtime", "project runtime", root, strict=False)
    edgetam_root = runtime_root / "segmentation" / "EdgeTAM"
    segmentation_python = _required_file(
        runtime_root / "segmentation" / ".venv" / "Scripts" / "python.exe",
        "segmentation worker Python",
        root,
        allow_missing=allow_missing_resources,
    )
    renderer_python = _required_file(
        runtime_root / "renderer" / ".venv" / "Scripts" / "python.exe",
        "renderer worker Python",
        root,
        allow_missing=allow_missing_resources,
    )
    model_config = _required_file(
        edgetam_root / "sam2" / "configs" / "edgetam.yaml",
        "EdgeTAM model config",
        root,
        allow_missing=allow_missing_resources,
    )
    checkpoint = _required_file(
        edgetam_root / "checkpoints" / "edgetam.pt",
        "EdgeTAM checkpoint",
        root,
        allow_missing=allow_missing_resources,
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
        "available_vram_limit_mb": load_user_vram_limit(
            runtime_root / "user-settings.json"
        ),
    }


def prepare_desktop_runtime(
    repo_root: Path,
    *,
    validate: bool = True,
    allow_missing_resources: bool = False,
) -> Path:
    try:
        root = repo_root.resolve(strict=True)
    except OSError as error:
        raise DesktopRuntimeError(
            f"repository root is unavailable: {repo_root.absolute()}"
        ) from error
    if not root.is_dir():
        raise DesktopRuntimeError(f"repository root is unavailable: {root}")
    payload = _runtime_payload(root, allow_missing_resources=allow_missing_resources)
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
            load_runtime_config(
                runtime_path,
                allow_missing_resources=allow_missing_resources,
            )
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
    parser.add_argument("--allow-missing-resources", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = prepare_desktop_runtime(
            args.repo_root,
            allow_missing_resources=args.allow_missing_resources,
        )
    except DesktopRuntimeError as error:
        raise SystemExit(str(error)) from error
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
