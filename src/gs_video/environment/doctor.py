from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import which as find_command
from typing import Any

from pydantic import BaseModel

from gs_video.domain.contracts import SegmentationBackend
from gs_video.segmentation.paths import has_reparse_component, worker_path


class EnvironmentIssue(BaseModel):
    code: str
    message: str


class EnvironmentReport(BaseModel):
    ready: bool
    vram_mb: int
    vram_limit_mb: int = 8192
    issues: list[EnvironmentIssue]
    renderer_versions: dict[str, str] | None = None


def probe_cuda() -> tuple[bool, int]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return False, 0

    if not torch.cuda.is_available():
        return False, 0

    total_vram = torch.cuda.get_device_properties(0).total_memory
    return True, total_vram // (1024 * 1024)


def probe_renderer() -> tuple[str | None, str | None]:
    """Import optional renderer dependencies independently and return identities."""

    torch_version: str | None = None
    gsplat_version: str | None = None
    try:
        import torch
    except ImportError:
        pass
    else:
        value = getattr(torch, "__version__", None)
        if isinstance(value, str) and value:
            torch_version = value
    try:
        import gsplat  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        value = getattr(gsplat, "__version__", None)
        if isinstance(value, str) and value:
            gsplat_version = value
    return torch_version, gsplat_version


class EnvironmentDoctor:
    def __init__(
        self,
        which: Callable[[str], str | None] = find_command,
        cuda_probe: Callable[[], tuple[bool, int]] = probe_cuda,
        *,
        segmentation_backend: SegmentationBackend | None = None,
        worker_prefix: Sequence[str] | None = None,
        model_config: Path | None = None,
        checkpoint: Path | None = None,
        process_runner: Callable[..., Any] = subprocess.run,
        check_renderer: bool = False,
        renderer_probe: Callable[[], tuple[str | None, str | None]] = probe_renderer,
        vram_limit_mb: int = 8192,
    ) -> None:
        self._which = which
        self._cuda_probe = cuda_probe
        self._segmentation_backend = segmentation_backend
        self._worker_prefix = tuple(worker_prefix or ())
        self._model_config = model_config
        self._checkpoint = checkpoint
        self._process_runner = process_runner
        self._check_renderer = check_renderer
        self._renderer_probe = renderer_probe
        self._vram_limit_mb = vram_limit_mb

    def _renderer_versions(self, issues: list[EnvironmentIssue]) -> dict[str, str] | None:
        if not self._check_renderer:
            return None
        torch_version, gsplat_version = self._renderer_probe()
        if torch_version is None:
            issues.append(EnvironmentIssue(code="torch_missing", message="PyTorch is not installed"))
        if gsplat_version is None:
            issues.append(EnvironmentIssue(code="gsplat_missing", message="gsplat is not installed"))
        elif re.match(r"^1(?:\.|$)", gsplat_version) is None:
            issues.append(
                EnvironmentIssue(code="gsplat_unsupported", message="gsplat 1.x is required")
            )
        if torch_version is None or gsplat_version is None:
            return None
        return {"torch": torch_version, "gsplat": gsplat_version}

    @staticmethod
    def _readable_file(path: Path | None) -> bool:
        if path is None or has_reparse_component(path) or not path.is_file():
            return False
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except OSError:
            return False
        return True

    def _check_segmentation(self, issues: list[EnvironmentIssue]) -> None:
        configured = any(
            value is not None
            for value in (self._segmentation_backend, self._model_config, self._checkpoint)
        ) or bool(self._worker_prefix)
        if not configured:
            return
        if self._segmentation_backend is None:
            issues.append(EnvironmentIssue(code="segmentation_backend_missing", message="分割后端未配置"))
        if not self._worker_prefix or any(not item for item in self._worker_prefix):
            issues.append(EnvironmentIssue(code="segmentation_worker_missing", message="分割 worker 命令未配置"))
        elif (
            not Path(self._worker_prefix[0]).is_file()
            and self._which(self._worker_prefix[0]) is None
        ):
            issues.append(
                EnvironmentIssue(code="segmentation_worker_missing", message="分割 worker 命令不存在")
            )
        if not self._readable_file(self._model_config):
            issues.append(
                EnvironmentIssue(code="segmentation_config_unreadable", message="分割模型配置不可读")
            )
        if not self._readable_file(self._checkpoint):
            issues.append(
                EnvironmentIssue(
                    code="segmentation_checkpoint_unreadable", message="分割模型 checkpoint 不可读"
                )
            )
        if any(issue.code.startswith("segmentation_") for issue in issues):
            return
        assert self._segmentation_backend is not None
        assert self._model_config is not None
        assert self._checkpoint is not None
        try:
            config_arg = worker_path(self._model_config, self._worker_prefix)
            checkpoint_arg = worker_path(self._checkpoint, self._worker_prefix)
        except ValueError:
            issues.append(
                EnvironmentIssue(code="segmentation_path_invalid", message="分割 worker 路径无效")
            )
            return
        command = [
            *self._worker_prefix,
            "-m", "gs_video.segmentation.worker", "--probe",
            "--backend", self._segmentation_backend.value,
            "--config", config_arg,
            "--checkpoint", checkpoint_arg,
        ]
        options: dict[str, object] = {
            "capture_output": True, "text": True, "shell": False, "timeout": 30
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            completed = self._process_runner(command, **options)
            lines = completed.stdout.splitlines()
            payload = json.loads(lines[0]) if len(lines) == 1 else None
            expected = {
                "type": "probe", "backend": self._segmentation_backend.value,
                "config": config_arg,
                "checkpoint": checkpoint_arg,
            }
            if (
                completed.returncode != 0
                or not isinstance(payload, dict)
                or set(payload) != {*expected, "predictor"}
                or any(payload[key] != value for key, value in expected.items())
                or not isinstance(payload["predictor"], str)
                or not payload["predictor"]
            ):
                raise ValueError("probe mismatch")
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            issues.append(
                EnvironmentIssue(code="segmentation_probe_failed", message="分割 worker 探测失败")
            )

    def check(self) -> EnvironmentReport:
        issues: list[EnvironmentIssue] = []

        if self._which("ffmpeg") is None:
            issues.append(
                EnvironmentIssue(code="ffmpeg_missing", message="ffmpeg command was not found")
            )
        if self._which("ffprobe") is None:
            issues.append(
                EnvironmentIssue(code="ffprobe_missing", message="ffprobe command was not found")
            )

        cuda_available, vram_mb = self._cuda_probe()
        if not cuda_available:
            issues.append(
                EnvironmentIssue(code="cuda_unavailable", message="CUDA is not available")
            )

        renderer_versions = self._renderer_versions(issues)
        self._check_segmentation(issues)

        return EnvironmentReport(
            ready=not issues,
            vram_mb=vram_mb,
            vram_limit_mb=self._vram_limit_mb,
            issues=issues,
            renderer_versions=renderer_versions,
        )
