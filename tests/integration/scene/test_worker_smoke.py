from __future__ import annotations

import os
from pathlib import Path

import pytest

from gs_video.pipeline.cancellation import CancellationToken
from gs_video.scene.worker_client import RendererWorkerClient
from gs_video.scene.worker_protocol import ProbeRequest


@pytest.mark.gpu
def test_configured_project_local_renderer_worker_probe() -> None:
    executable = os.environ.get("GS_VIDEO_RENDERER_PYTHON")
    project_root = os.environ.get("GS_VIDEO_PROJECT_ROOT")
    if not executable or not project_root:
        pytest.fail(
            "GPU release gate requires GS_VIDEO_RENDERER_PYTHON and "
            "GS_VIDEO_PROJECT_ROOT; it is intentionally not skipped"
        )
    executable_path = Path(executable).absolute()
    root = Path(project_root).absolute()
    if root not in executable_path.parents:
        pytest.fail("renderer Python must be installed inside the project workspace")
    client = RendererWorkerClient(
        worker_prefix=(str(executable_path),),
        log_path=root / "logs" / "renderer-worker-smoke.log",
    )
    identity = client.probe(ProbeRequest(type="probe"), CancellationToken())
    assert identity.device == "cuda"
    assert identity.torch and identity.gsplat
