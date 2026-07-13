from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gs_video.domain.contracts import RenderSettings
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.gsplat_renderer import GsplatRenderer
from gs_video.scene.ply import load_gaussian_ply


@pytest.mark.gpu
def test_real_gsplat_tiny_scene_writes_png_below_one_gibibyte(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not configured")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    # This opt-in smoke owns its isolated process and may reset peak stats. Production never does.
    torch.cuda.reset_peak_memory_stats()
    scene = load_gaussian_ply(Path("tests/fixtures/scene/tiny_gaussians.ply"))
    # Center the first fixture Gaussian at positive camera depth so visibility is deterministic.
    camera = OrbitCamera((1.0, 2.0, 3.0), 3.0, 0.0, 0.0, 60.0)
    background = (0.05, 0.1, 0.15)
    result = GsplatRenderer().render(
        scene, [camera], tmp_path / "frames",
        RenderSettings(width=64, height=36, sh_degree=0, background=background),
        lambda *_: None, CancellationToken(),
    )
    torch.cuda.synchronize()
    with Image.open(result.frame_paths[0]) as image:
        image.load()
        assert image.mode == "RGB"
        assert image.size == (64, 36)
        pixels = np.asarray(image)
    background_rgb = np.rint(np.asarray(background) * 255).astype(np.uint8)
    assert np.any(pixels != background_rgb)
    assert result.frame_paths[0].stat().st_size > 0
    assert torch.cuda.max_memory_allocated() - baseline < 1024**3
