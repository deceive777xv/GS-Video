from __future__ import annotations

from collections.abc import Callable
from shutil import which as find_command

from pydantic import BaseModel


class EnvironmentIssue(BaseModel):
    code: str
    message: str


class EnvironmentReport(BaseModel):
    ready: bool
    vram_mb: int
    issues: list[EnvironmentIssue]


def probe_cuda() -> tuple[bool, int]:
    try:
        import torch
    except ImportError:
        return False, 0

    if not torch.cuda.is_available():
        return False, 0

    total_vram = torch.cuda.get_device_properties(0).total_memory
    return True, total_vram // (1024 * 1024)


class EnvironmentDoctor:
    def __init__(
        self,
        which: Callable[[str], str | None] = find_command,
        cuda_probe: Callable[[], tuple[bool, int]] = probe_cuda,
    ) -> None:
        self._which = which
        self._cuda_probe = cuda_probe

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

        return EnvironmentReport(ready=not issues, vram_mb=vram_mb, issues=issues)
