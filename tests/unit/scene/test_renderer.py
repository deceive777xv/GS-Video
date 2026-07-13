from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gs_video.domain.contracts import RenderSettings
from gs_video.domain.errors import CancelledError, GsVideoError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.gsplat_renderer import (
    GsplatRasterizerAdapter,
    GsplatRenderer,
    _load_gsplat_adapter,
)
from gs_video.scene.ply import GaussianScene, estimate_scene_vram


def tiny_scene(*, coefficients: int = 4) -> GaussianScene:
    return GaussianScene(
        means=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        scales=np.array([[0.0, np.log(2.0), np.log(3.0)]], dtype=np.float32),
        quats=np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.0], dtype=np.float32),
        colors=np.arange(coefficients * 3, dtype=np.float32).reshape(1, coefficients, 3),
    )


def camera(*, yaw: float = 0.0) -> OrbitCamera:
    return OrbitCamera(
        target=(0.0, 0.0, 1.0), distance=1.0, yaw=yaw, pitch=0.0, fov_y_degrees=60.0
    )


class RecordingRasterizer:
    version = "1.5.3"

    def __init__(self, *, token: CancellationToken | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.token = token

    def __call__(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        self.calls.append(kwargs)
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        channels = 4 if kwargs["render_mode"] == "RGB+ED" else 3
        render = np.full((1, height, width, channels), 0.25, dtype=np.float32)
        if channels == 4:
            render[..., 3] = 2.5
        alpha = np.ones((1, height, width, 1), dtype=np.float32)
        if self.token is not None:
            self.token.cancel()
        return render, alpha, {}


def settings(**overrides: object) -> RenderSettings:
    values: dict[str, object] = {
        "width": 64,
        "height": 36,
        "sh_degree": 1,
        "background": (0.0, 0.0, 0.0),
        "preview_stride": 1,
    }
    values.update(overrides)
    return RenderSettings(**values)  # type: ignore[arg-type]


def test_preview_settings_default_to_height_540_and_validate_inputs() -> None:
    assert RenderSettings(width=960).height == 540
    with pytest.raises(ValueError, match="preview_stride"):
        settings(preview_stride=0)
    with pytest.raises(ValueError, match="finite"):
        settings(background=(0.0, float("nan"), 0.0))
    with pytest.raises(ValueError, match="positive"):
        settings(width=0)


def test_renderer_converts_raw_parameters_and_uses_world_to_camera_matrices(
    tmp_path: Path,
) -> None:
    rasterizer = RecordingRasterizer()
    selected_camera = camera(yaw=20.0)
    renderer = GsplatRenderer(rasterizer=rasterizer, device="cpu")

    renderer.render(
        tiny_scene(), [selected_camera], tmp_path / "frames", settings(), lambda *_: None,
        CancellationToken(),
    )

    call = rasterizer.calls[0]
    np.testing.assert_allclose(call["quats"], [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(call["scales"], [[1.0, 2.0, 3.0]], rtol=1e-6)
    np.testing.assert_allclose(call["opacities"], [0.5])
    np.testing.assert_array_equal(call["colors"], tiny_scene().colors)
    np.testing.assert_allclose(call["viewmats"], selected_camera.view_matrix()[None], atol=1e-6)
    np.testing.assert_allclose(call["Ks"], selected_camera.intrinsics(64, 36)[None])
    assert np.asarray(call["viewmats"]).shape == (1, 4, 4)
    assert np.asarray(call["Ks"]).shape == (1, 3, 3)


def test_renderer_rejects_nonfinite_passthrough_scene_fields_before_rasterizing(
    tmp_path: Path,
) -> None:
    scene = tiny_scene()
    scene.means[0, 0] = np.nan
    rasterizer = RecordingRasterizer()
    with pytest.raises(ValueError, match="finite"):
        GsplatRenderer(rasterizer=rasterizer, device="cpu").render(
            scene, [camera()], tmp_path / "frames", settings(), lambda *_: None,
            CancellationToken(),
        )
    assert rasterizer.calls == []


def test_renderer_sigmoid_is_stable_for_extreme_opacity_logits(tmp_path: Path) -> None:
    scene = tiny_scene()
    scene.opacities[0] = -1000.0
    rasterizer = RecordingRasterizer()
    with np.errstate(over="raise"):
        GsplatRenderer(rasterizer=rasterizer, device="cpu").render(
            scene, [camera()], tmp_path / "frames", settings(), lambda *_: None,
            CancellationToken(),
        )
    assert np.asarray(rasterizer.calls[0]["opacities"])[0] == pytest.approx(0.0)


def test_renderer_reports_scale_activation_overflow_as_validation_error(tmp_path: Path) -> None:
    scene = tiny_scene()
    scene.scales[0, 0] = 1000.0
    with np.errstate(over="raise"):
        with pytest.raises(ValueError, match="finite"):
            GsplatRenderer(rasterizer=RecordingRasterizer(), device="cpu").render(
                scene, [camera()], tmp_path / "frames", settings(), lambda *_: None,
                CancellationToken(),
            )


def test_renderer_calls_one_camera_at_a_time_and_serializes_preview_stride(
    tmp_path: Path,
) -> None:
    rasterizer = RecordingRasterizer()
    events: list[tuple[int, int, str]] = []
    output = tmp_path / "frames"
    result = GsplatRenderer(rasterizer=rasterizer, device="cpu").render(
        tiny_scene(), [camera(), camera(yaw=10), camera(yaw=20)], output,
        settings(preview_stride=2), lambda *event: events.append(event), CancellationToken(),
    )

    assert [np.asarray(call["viewmats"]).shape[0] for call in rasterizer.calls] == [1, 1]
    assert result.frame_count == 2
    assert result.source_frame_indices == (0, 2)
    assert result.frame_paths == (output / "000001.png", output / "000003.png")
    assert events == [(1, 2, "渲染背景 1/2"), (2, 2, "渲染背景 2/2")]
    with Image.open(result.frame_paths[0]) as image:
        assert image.mode == "RGB"
        assert image.size == (64, 36)
        assert image.getpixel((0, 0)) == (64, 64, 64)


def test_renderer_rejects_sh_degree_not_present_in_scene_before_rasterizing(
    tmp_path: Path,
) -> None:
    rasterizer = RecordingRasterizer()
    with pytest.raises(ValueError, match="SH degree"):
        GsplatRenderer(rasterizer=rasterizer, device="cpu").render(
            tiny_scene(coefficients=1), [camera()], tmp_path / "frames",
            settings(sh_degree=1), lambda *_: None, CancellationToken(),
        )
    assert rasterizer.calls == []


def test_renderer_enforces_eighty_percent_vram_policy_before_work(tmp_path: Path) -> None:
    scene = tiny_scene()
    required = estimate_scene_vram(scene, 64, 36)
    rasterizer = RecordingRasterizer()
    renderer = GsplatRenderer(
        rasterizer=rasterizer, device="cpu", available_vram_bytes=required
    )
    with pytest.raises(GsVideoError, match="VRAM"):
        renderer.render(
            scene, [camera()], tmp_path / "frames", settings(), lambda *_: None,
            CancellationToken(),
        )
    assert rasterizer.calls == []


def test_cancellation_after_rasterizer_releases_frame_and_does_not_publish(tmp_path: Path) -> None:
    token = CancellationToken()
    output = tmp_path / "frames"
    renderer = GsplatRenderer(rasterizer=RecordingRasterizer(token=token), device="cpu")
    with pytest.raises(CancelledError):
        renderer.render(
            tiny_scene(), [camera()], output, settings(), lambda *_: None, token
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".frames.staging-*"))


def test_render_error_preserves_previous_valid_directory_and_cleans_staging(
    tmp_path: Path,
) -> None:
    output = tmp_path / "frames"
    output.mkdir()
    (output / "old.png").write_bytes(b"previous")

    class InvalidSecondFrame(RecordingRasterizer):
        def __call__(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
            result = super().__call__(**kwargs)
            if len(self.calls) == 2:
                result[0][0, 0, 0, 0] = np.nan
            return result

    with pytest.raises(GsVideoError, match="finite"):
        GsplatRenderer(rasterizer=InvalidSecondFrame(), device="cpu").render(
            tiny_scene(), [camera(), camera()], output, settings(), lambda *_: None,
            CancellationToken(),
        )
    assert (output / "old.png").read_bytes() == b"previous"
    assert not list(tmp_path.glob(".frames.staging-*"))
    assert not list(tmp_path.glob(".frames.backup-*"))


@pytest.mark.parametrize(
    ("render_shape", "alpha_shape", "message"),
    [
        ((1, 36, 64, 2), (1, 36, 64, 1), "shape"),
        ((1, 36, 64, 3), (1, 36, 64, 2), "alpha"),
    ],
)
def test_renderer_validates_rasterizer_outputs(
    tmp_path: Path, render_shape: tuple[int, ...], alpha_shape: tuple[int, ...], message: str
) -> None:
    def invalid(**kwargs: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        del kwargs
        return (
            np.zeros(render_shape, dtype=np.float32),
            np.zeros(alpha_shape, dtype=np.float32),
            {},
        )

    invalid.version = "1.5.3"  # type: ignore[attr-defined]
    with pytest.raises(GsVideoError, match=message):
        GsplatRenderer(rasterizer=invalid, device="cpu").render(
            tiny_scene(), [camera()], tmp_path / "frames", settings(), lambda *_: None,
            CancellationToken(),
        )


def test_renderer_rejects_non_float32_rasterizer_outputs(tmp_path: Path) -> None:
    def float64_output(**kwargs: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        height, width = int(kwargs["height"]), int(kwargs["width"])
        return (
            np.zeros((1, height, width, 3), dtype=np.float64),
            np.ones((1, height, width, 1), dtype=np.float32),
            {},
        )

    float64_output.version = "1.5.3"  # type: ignore[attr-defined]
    with pytest.raises(GsVideoError, match="float32"):
        GsplatRenderer(rasterizer=float64_output, device="cpu").render(
            tiny_scene(), [camera()], tmp_path / "frames", settings(), lambda *_: None,
            CancellationToken(),
        )

def test_render_pick_returns_only_rgb_and_expected_depth_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rasterizer = RecordingRasterizer()
    pick = GsplatRenderer(rasterizer=rasterizer, device="cpu").render_pick(
        tiny_scene(), camera(), width=64, height=36
    )
    assert pick.rgb.shape == (36, 64, 3)
    assert pick.rgb.dtype == np.uint8
    assert pick.expected_depth.shape == (36, 64)
    assert pick.expected_depth.dtype == np.float32
    assert np.all(pick.expected_depth == 2.5)
    assert rasterizer.calls[0]["render_mode"] == "RGB+ED"
    assert list(tmp_path.iterdir()) == []


def test_renderer_rejects_alpha_outside_unit_interval_and_negative_pick_depth(
    tmp_path: Path,
) -> None:
    class InvalidSemanticRasterizer(RecordingRasterizer):
        def __init__(self, *, invalid_depth: bool) -> None:
            super().__init__()
            self.invalid_depth = invalid_depth

        def __call__(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
            render, alpha, meta = super().__call__(**kwargs)
            if self.invalid_depth:
                render[..., 3] = -1.0
            else:
                alpha[...] = 1.1
            return render, alpha, meta

    with pytest.raises(GsVideoError, match="alpha"):
        GsplatRenderer(
            rasterizer=InvalidSemanticRasterizer(invalid_depth=False), device="cpu"
        ).render(
            tiny_scene(), [camera()], tmp_path / "alpha", settings(), lambda *_: None,
            CancellationToken(),
        )
    with pytest.raises(GsVideoError, match="depth"):
        GsplatRenderer(
            rasterizer=InvalidSemanticRasterizer(invalid_depth=True), device="cpu"
        ).render_pick(tiny_scene(), camera(), 64, 36)


def test_gsplat_adapter_selects_supported_sh_keyword_and_rejects_other_versions() -> None:
    calls: list[dict[str, object]] = []

    def current(*, colors_sh_degree: int, **kwargs: object) -> tuple[object, object, object]:
        calls.append({"colors_sh_degree": colors_sh_degree, **kwargs})
        return object(), object(), {}

    adapter = GsplatRasterizerAdapter(
        rasterization=current, torch_module=None, device="cpu", version="1.5.3"
    )
    adapter(sh_degree=2)
    assert calls[0]["colors_sh_degree"] == 2
    assert "sh_degree" not in calls[0]
    assert "colors_sh_degree" in inspect.signature(current).parameters

    with pytest.raises(GsVideoError, match="gsplat 1.x"):
        GsplatRasterizerAdapter(
            rasterization=current, torch_module=None, device="cpu", version="2.0.0"
        )


def test_gsplat_adapter_supports_legacy_1x_sh_keyword() -> None:
    calls: list[dict[str, object]] = []

    def legacy(*, sh_degree: int, **kwargs: object) -> tuple[object, object, object]:
        calls.append({"sh_degree": sh_degree, **kwargs})
        return object(), object(), {}

    GsplatRasterizerAdapter(
        rasterization=legacy, torch_module=None, device="cpu", version="1.0.0"
    )(sh_degree=1)
    assert calls == [{"sh_degree": 1}]


def test_gsplat_adapter_runs_rasterization_under_inference_mode() -> None:
    entered: list[bool] = []

    @contextmanager
    def inference_mode() -> object:
        entered.append(True)
        yield

    torch = SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, **kwargs: np.asarray(value, dtype=kwargs["dtype"]),
        inference_mode=inference_mode,
    )

    def rasterization(*, sh_degree: int, **kwargs: object) -> tuple[object, object, object]:
        del sh_degree, kwargs
        return object(), object(), {}

    GsplatRasterizerAdapter(
        rasterization=rasterization, torch_module=torch, device="cpu", version="1.0.0"
    )(sh_degree=0, means=np.zeros((1, 3), dtype=np.float32))
    assert entered == [True]


def test_lazy_loader_accepts_rendering_module_export(monkeypatch: pytest.MonkeyPatch) -> None:
    def rasterization(*, colors_sh_degree: int, **kwargs: object) -> tuple[object, object, object]:
        del colors_sh_degree, kwargs
        return object(), object(), {}

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float32=np.float32,
    )
    fake_gsplat = SimpleNamespace(
        __version__="1.5.3", rendering=SimpleNamespace(rasterization=rasterization)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)
    adapter, _metrics = _load_gsplat_adapter("cuda")
    assert adapter.version == "1.5.3"
