from __future__ import annotations

import inspect
import re
import shutil
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from gs_video.domain.contracts import PickBuffer, RenderSequence, RenderSettings
from gs_video.domain.errors import GsVideoError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.scene.camera import CameraMatrices
from gs_video.scene.ply import GaussianScene, assess_scene_vram
from gs_video.segmentation.paths import has_reparse_component


Rasterizer = Callable[..., tuple[object, object, object]]


@dataclass(frozen=True)
class PreparedPreviewScene:
    runtime: dict[str, object]
    rasterizer: Rasterizer
    metrics: CudaMetrics
    sh_degree: int
    maximum_width: int
    maximum_height: int


@dataclass(frozen=True)
class PreparedPreviewTimings:
    gpu_raster_ms: float
    readback_ms: float


class CudaMetrics(Protocol):
    def available_bytes(self) -> int: ...

    def peak_allocated_bytes(self) -> int | None: ...

    def release_frame(self) -> None: ...


class _CpuMetrics:
    def available_bytes(self) -> int:
        return sys.maxsize

    def peak_allocated_bytes(self) -> int | None:
        return None

    def release_frame(self) -> None:
        return


class _TorchCudaMetrics:
    def __init__(self, torch_module: Any, device: str) -> None:
        self._torch = torch_module
        self._device = device

    def available_bytes(self) -> int:
        free, _total = self._torch.cuda.mem_get_info(self._device)
        return int(free)

    def peak_allocated_bytes(self) -> int | None:
        if not self._torch.cuda.is_available():
            return None
        return int(self._torch.cuda.max_memory_allocated(self._device))

    def release_frame(self) -> None:
        # Synchronize the current device so frame-owned CUDA work is complete. Deliberately
        # do not reset global peak stats or empty the allocator, both of which affect peers.
        self._torch.cuda.synchronize(self._device)


class GsplatRasterizerAdapter:
    """Version-checked adapter for the supported gsplat 1.x rasterization API."""

    def __init__(
        self,
        *,
        rasterization: Rasterizer,
        torch_module: Any,
        device: str,
        version: str,
    ) -> None:
        if re.match(r"^1(?:\.|$)", version) is None:
            raise GsVideoError(f"需要 gsplat 1.x，当前版本为 {version!r}")
        parameters = inspect.signature(rasterization).parameters
        if "colors_sh_degree" in parameters:
            self._sh_keyword = "colors_sh_degree"
        elif "sh_degree" in parameters:
            self._sh_keyword = "sh_degree"
        else:
            raise GsVideoError("不支持的 gsplat 1.x rasterization SH 参数签名")
        self._rasterization = rasterization
        self._torch = torch_module
        self._device = device
        self.version = version

    def __call__(self, **kwargs: object) -> tuple[object, object, object]:
        sh_degree = kwargs.pop("sh_degree")
        kwargs[self._sh_keyword] = sh_degree
        if self._torch is not None:
            for name in (
                "means", "quats", "scales", "opacities", "colors", "viewmats", "Ks",
                "backgrounds",
            ):
                if name in kwargs:
                    kwargs[name] = self._torch.as_tensor(
                        kwargs[name], dtype=self._torch.float32, device=self._device
                    )
        if self._torch is None:
            return self._rasterization(**kwargs)
        with self._torch.inference_mode():
            return self._rasterization(**kwargs)

    def prepare_scene(self, runtime: dict[str, object]) -> dict[str, object]:
        if self._torch is None:
            return dict(runtime)
        return {
            name: self._torch.as_tensor(
                value, dtype=self._torch.float32, device=self._device
            )
            for name, value in runtime.items()
        }


def _load_gsplat_adapter(device: str) -> tuple[GsplatRasterizerAdapter, _TorchCudaMetrics]:
    try:
        import torch  # type: ignore[import-not-found]
        import gsplat  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GsVideoError("renderer 需要可选依赖 torch 与 gsplat") from exc
    version = getattr(gsplat, "__version__", None)
    rasterization = getattr(gsplat, "rasterization", None)
    if not callable(rasterization):
        rendering = getattr(gsplat, "rendering", None)
        rasterization = getattr(rendering, "rasterization", None)
    if not isinstance(version, str) or not version or not callable(rasterization):
        raise GsVideoError("gsplat 缺少可用的版本身份或 rasterization API")
    if not torch.cuda.is_available():
        raise GsVideoError("renderer 需要可用的 CUDA device")
    adapter = GsplatRasterizerAdapter(
        rasterization=rasterization, torch_module=torch, device=device, version=version
    )
    return adapter, _TorchCudaMetrics(torch, device)


def _to_numpy(value: object) -> np.ndarray:
    result = value
    detach = getattr(result, "detach", None)
    if callable(detach):
        result = detach()
    cpu = getattr(result, "cpu", None)
    if callable(cpu):
        result = cpu()
    numpy_method = getattr(result, "numpy", None)
    if callable(numpy_method):
        result = numpy_method()
    return np.asarray(result)


def _runtime_scene(scene: GaussianScene) -> dict[str, npt.NDArray[np.float32]]:
    if not all(
        np.isfinite(array).all()
        for array in (scene.means, scene.scales, scene.quats, scene.opacities, scene.colors)
    ):
        raise ValueError("scene parameters must contain only finite values")
    norms = np.linalg.norm(scene.quats, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("scene quaternions must have finite nonzero norm")
    with np.errstate(over="ignore", invalid="ignore"):
        scales = np.exp(scene.scales).astype(np.float32, copy=False)
    opacities = np.empty_like(scene.opacities)
    nonnegative = scene.opacities >= 0
    opacities[nonnegative] = 1.0 / (1.0 + np.exp(-scene.opacities[nonnegative]))
    exponentials = np.exp(scene.opacities[~nonnegative])
    opacities[~nonnegative] = exponentials / (1.0 + exponentials)
    if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(opacities)):
        raise ValueError("scene runtime parameters must remain finite")
    return {
        "means": scene.means,
        "quats": np.ascontiguousarray(scene.quats / norms, dtype=np.float32),
        "scales": np.ascontiguousarray(scales, dtype=np.float32),
        "opacities": np.ascontiguousarray(opacities, dtype=np.float32),
        "colors": scene.colors,
    }


class GsplatRenderer:
    def __init__(
        self,
        *,
        rasterizer: Rasterizer | None = None,
        device: str = "cuda",
        available_vram_bytes: int | None = None,
        cuda_metrics: CudaMetrics | None = None,
        backup_cleanup: Callable[[Path], None] | None = None,
    ) -> None:
        self._device = device
        self._available_vram_bytes = available_vram_bytes
        self._configured_rasterizer = rasterizer
        self._configured_metrics = cuda_metrics
        self._dependency_pair: tuple[Rasterizer, CudaMetrics] | None = None
        self._dependency_lock = Lock()
        self._operation_lock = RLock()
        self._backup_cleanup = backup_cleanup or shutil.rmtree

    def _dependencies(self) -> tuple[Rasterizer, CudaMetrics]:
        pair = self._dependency_pair
        if pair is not None:
            return pair
        with self._dependency_lock:
            pair = self._dependency_pair
            if pair is None:
                rasterizer = self._configured_rasterizer
                metrics = self._configured_metrics
                if rasterizer is None:
                    rasterizer, loaded_metrics = _load_gsplat_adapter(self._device)
                    if metrics is None:
                        metrics = loaded_metrics
                elif metrics is None:
                    metrics = _CpuMetrics()
                pair = (rasterizer, metrics)
                self._dependency_pair = pair
        return pair

    @staticmethod
    def _version(rasterizer: Rasterizer) -> str:
        value = getattr(rasterizer, "version", None)
        return value if isinstance(value, str) and value else "injected-fake"

    @staticmethod
    def _validate_scene_settings(scene: GaussianScene, settings: RenderSettings) -> None:
        required_coefficients = (settings.sh_degree + 1) ** 2
        if scene.colors.shape[1] < required_coefficients:
            raise ValueError(
                f"SH degree {settings.sh_degree} requires {required_coefficients} coefficients"
            )

    @staticmethod
    def _scene_sh_degree(scene: GaussianScene) -> int:
        degrees = {1: 0, 4: 1, 9: 2, 16: 3}
        try:
            return degrees[int(scene.colors.shape[1])]
        except KeyError as exc:
            raise ValueError("scene SH coefficient count must be exactly 1, 4, 9, or 16") from exc

    @staticmethod
    def _camera_arguments(
        camera: CameraMatrices, width: int, height: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        try:
            view = np.asarray(camera.view_matrix(), dtype=np.float32)
            intrinsics = np.asarray(camera.intrinsics(width, height), dtype=np.float32)
        except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError("camera must provide valid view matrix and intrinsics") from exc
        if view.shape != (4, 4) or intrinsics.shape != (3, 3) or not (
            np.isfinite(view).all() and np.isfinite(intrinsics).all()
        ):
            raise ValueError("camera matrices must have valid finite shapes")
        return view[None], intrinsics[None]

    @staticmethod
    def _validate_outputs(
        render: object,
        alpha: object,
        *,
        width: int,
        height: int,
        channels: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        render_array = _to_numpy(render)
        alpha_array = _to_numpy(alpha)
        if render_array.shape != (1, height, width, channels):
            raise GsVideoError(
                f"rasterizer render shape 无效: {render_array.shape}, expected "
                f"{(1, height, width, channels)}"
            )
        if alpha_array.shape != (1, height, width, 1):
            raise GsVideoError(
                f"rasterizer alpha shape 无效: {alpha_array.shape}, expected "
                f"{(1, height, width, 1)}"
            )
        if render_array.dtype != np.float32 or alpha_array.dtype != np.float32:
            raise GsVideoError("rasterizer outputs must use float32 dtype")
        if not np.isfinite(render_array).all() or not np.isfinite(alpha_array).all():
            raise GsVideoError("rasterizer outputs must contain only finite values")
        if np.any(alpha_array < 0.0) or np.any(alpha_array > 1.0):
            raise GsVideoError("rasterizer alpha must remain in the [0, 1] interval")
        return render_array, alpha_array

    @staticmethod
    def _rgb8(render: np.ndarray) -> npt.NDArray[np.uint8]:
        return np.asarray(np.rint(np.clip(render[0, ..., :3], 0.0, 1.0) * 255), dtype=np.uint8)

    @staticmethod
    def _validate_output(output_dir: Path) -> Path:
        output = Path(output_dir).absolute()
        if not output.name or output == output.parent:
            raise GsVideoError("render output directory is unsafe")
        if has_reparse_component(output.parent) or has_reparse_component(output):
            raise GsVideoError("render output directory must not contain links or reparse points")
        if output.exists() and not output.is_dir():
            raise GsVideoError("render output path must be a directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        GsplatRenderer._assert_safe_output(output)
        return output

    @staticmethod
    def _assert_safe_output(output: Path) -> None:
        if has_reparse_component(output.parent) or has_reparse_component(output):
            raise GsVideoError("render output directory must not contain links or reparse points")
        if output.exists() and not output.is_dir():
            raise GsVideoError("render output path must be a directory")

    @staticmethod
    def _assert_owned_tree(path: Path, parent: Path, prefix: str) -> None:
        absolute = path.absolute()
        expected_parent = parent.absolute()
        suffix = absolute.name.removeprefix(prefix)
        if (
            absolute.parent != expected_parent
            or not absolute.name.startswith(prefix)
            or len(suffix) != 32
            or any(character not in "0123456789abcdef" for character in suffix)
            or has_reparse_component(absolute)
        ):
            raise GsVideoError("refusing to remove unchecked renderer-owned directory")

    def _cleanup_staging_best_effort(self, staging: Path, output: Path) -> None:
        try:
            self._assert_owned_tree(staging, output.parent, f".{output.name}.staging-")
            shutil.rmtree(staging)
        except (GsVideoError, OSError):
            return

    def _publish(self, staging: Path, output: Path) -> None:
        backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
        moved_old = False
        try:
            if output.exists():
                output.replace(backup)
                moved_old = True
            staging.replace(output)
        except BaseException:
            if moved_old and backup.exists() and not output.exists():
                backup.replace(output)
            raise
        if moved_old:
            try:
                self._assert_owned_tree(backup, output.parent, f".{output.name}.backup-")
                self._backup_cleanup(backup)
            except Exception:
                # Publication is already complete. Preserve the checked backup for later
                # housekeeping rather than reporting a false render failure.
                return

    def _admit(
        self, scene: GaussianScene, width: int, height: int, metrics: CudaMetrics
    ) -> None:
        available = (
            self._available_vram_bytes
            if self._available_vram_bytes is not None
            else metrics.available_bytes()
        )
        if available <= 0:
            raise GsVideoError("available VRAM must be positive")
        assessment = assess_scene_vram(scene, width, height, available)
        if not assessment.accepted:
            raise GsVideoError(
                f"scene exceeds 80% VRAM policy: estimated {assessment.estimated_bytes} bytes"
            )

    def _call(
        self,
        rasterizer: Rasterizer,
        runtime: Mapping[str, object],
        camera: CameraMatrices,
        settings: RenderSettings,
        render_mode: str,
    ) -> tuple[object, object, object]:
        viewmats, intrinsics = self._camera_arguments(camera, settings.width, settings.height)
        arguments: dict[str, object] = {
            **runtime,
            "viewmats": viewmats,
            "Ks": intrinsics,
            "width": settings.width,
            "height": settings.height,
            "sh_degree": settings.sh_degree,
            "packed": True,
            "render_mode": render_mode,
        }
        if render_mode == "RGB":
            arguments["backgrounds"] = np.asarray(settings.background, dtype=np.float32)
        return rasterizer(
            **arguments,
        )

    def render(
        self,
        scene: GaussianScene,
        cameras: Sequence[CameraMatrices],
        output_dir: Path,
        settings: RenderSettings,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence:
        with self._operation_lock:
            return self._render_locked(scene, cameras, output_dir, settings, emit, token)

    def _render_locked(
        self,
        scene: GaussianScene,
        cameras: Sequence[CameraMatrices],
        output_dir: Path,
        settings: RenderSettings,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence:
        if not cameras:
            raise ValueError("camera list must not be empty")
        self._validate_scene_settings(scene, settings)
        output = self._validate_output(output_dir)
        rasterizer, metrics = self._dependencies()
        self._admit(scene, settings.width, settings.height, metrics)
        runtime = _runtime_scene(scene)
        selected = tuple(range(0, len(cameras), settings.preview_stride))
        staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
        self._assert_safe_output(output)
        staging.mkdir()
        paths: list[Path] = []
        try:
            for rendered_index, source_index in enumerate(selected, start=1):
                token.raise_if_cancelled()
                render: object | None = None
                alpha: object | None = None
                meta: object | None = None
                try:
                    render, alpha, meta = self._call(
                        rasterizer, runtime, cameras[source_index], settings, "RGB"
                    )
                    token.raise_if_cancelled()
                    render_array, _alpha_array = self._validate_outputs(
                        render, alpha, width=settings.width, height=settings.height, channels=3
                    )
                    token.raise_if_cancelled()
                    name = f"{source_index + 1:06d}.png"
                    Image.fromarray(self._rgb8(render_array)).save(staging / name)
                    paths.append(output / name)
                    token.raise_if_cancelled()
                finally:
                    del render, alpha, meta
                    metrics.release_frame()
                message = f"渲染背景 {rendered_index}/{len(selected)}"
                if rendered_index % 10 == 0:
                    peak = metrics.peak_allocated_bytes()
                    if peak is not None:
                        message += f"；CUDA 峰值显存 {peak // (1024 * 1024)} MiB"
                emit(rendered_index, len(selected), message)
            token.raise_if_cancelled()
            self._publish(staging, output)
        except BaseException:
            if staging.exists():
                self._cleanup_staging_best_effort(staging, output)
            raise
        return RenderSequence(
            frame_dir=output,
            frame_paths=tuple(paths),
            source_frame_indices=selected,
            width=settings.width,
            height=settings.height,
            implementation_version=f"gsplat-{self._version(rasterizer)}",
        )

    def render_pick(
        self, scene: GaussianScene, camera: CameraMatrices, width: int, height: int
    ) -> PickBuffer:
        with self._operation_lock:
            return self._render_pick_locked(scene, camera, width, height)

    def prepare_preview(
        self,
        scene: GaussianScene,
        *,
        width: int,
        height: int,
        sh_degree: int,
    ) -> PreparedPreviewScene:
        with self._operation_lock:
            effective_sh_degree = min(sh_degree, self._scene_sh_degree(scene))
            settings = RenderSettings(
                width=width, height=height, sh_degree=effective_sh_degree
            )
            self._validate_scene_settings(scene, settings)
            rasterizer, metrics = self._dependencies()
            self._admit(scene, width, height, metrics)
            runtime: dict[str, object] = dict(_runtime_scene(scene))
            prepare = getattr(rasterizer, "prepare_scene", None)
            if callable(prepare):
                runtime = dict(prepare(runtime))
            return PreparedPreviewScene(
                runtime=runtime,
                rasterizer=rasterizer,
                metrics=metrics,
                sh_degree=effective_sh_degree,
                maximum_width=width,
                maximum_height=height,
            )

    @staticmethod
    def _validate_prepared_size(
        prepared: PreparedPreviewScene, width: int, height: int
    ) -> None:
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or width > prepared.maximum_width
            or height > prepared.maximum_height
        ):
            raise ValueError("prepared preview dimensions exceed the admitted size")

    def render_prepared_rgb(
        self,
        prepared: PreparedPreviewScene,
        camera: CameraMatrices,
        width: int,
        height: int,
    ) -> npt.NDArray[np.uint8]:
        with self._operation_lock:
            self._validate_prepared_size(prepared, width, height)
            render: object | None = None
            alpha: object | None = None
            meta: object | None = None
            try:
                settings = RenderSettings(
                    width=width, height=height, sh_degree=prepared.sh_degree
                )
                render, alpha, meta = self._call(
                    prepared.rasterizer,
                    prepared.runtime,
                    camera,
                    settings,
                    "RGB",
                )
                render_array, alpha_array = self._validate_outputs(
                    render, alpha, width=width, height=height, channels=3
                )
                return np.ascontiguousarray(self._rgb8(render_array))
            finally:
                del render, alpha, meta
                prepared.metrics.release_frame()

    def render_prepared_rgb_profiled(
        self,
        prepared: PreparedPreviewScene,
        camera: CameraMatrices,
        width: int,
        height: int,
    ) -> tuple[npt.NDArray[np.uint8], PreparedPreviewTimings]:
        with self._operation_lock:
            self._validate_prepared_size(prepared, width, height)
            render: object | None = None
            alpha: object | None = None
            meta: object | None = None
            released = False
            try:
                settings = RenderSettings(
                    width=width, height=height, sh_degree=prepared.sh_degree
                )
                started = perf_counter()
                render, alpha, meta = self._call(
                    prepared.rasterizer,
                    prepared.runtime,
                    camera,
                    settings,
                    "RGB",
                )
                prepared.metrics.release_frame()
                released = True
                raster_done = perf_counter()
                render_array, alpha_array = self._validate_outputs(
                    render, alpha, width=width, height=height, channels=3
                )
                rgb = np.ascontiguousarray(self._rgb8(render_array))
                readback_done = perf_counter()
                return rgb, PreparedPreviewTimings(
                    gpu_raster_ms=(raster_done - started) * 1000,
                    readback_ms=(readback_done - raster_done) * 1000,
                )
            finally:
                del render, alpha, meta
                if not released:
                    prepared.metrics.release_frame()

    def render_prepared_pick(
        self,
        prepared: PreparedPreviewScene,
        camera: CameraMatrices,
        width: int,
        height: int,
    ) -> PickBuffer:
        with self._operation_lock:
            self._validate_prepared_size(prepared, width, height)
            settings = RenderSettings(
                width=width, height=height, sh_degree=prepared.sh_degree
            )
            render: object | None = None
            alpha: object | None = None
            meta: object | None = None
            try:
                render, alpha, meta = self._call(
                    prepared.rasterizer,
                    prepared.runtime,
                    camera,
                    settings,
                    "RGB+ED",
                )
                render_array, alpha_array = self._validate_outputs(
                    render, alpha, width=width, height=height, channels=4
                )
                if np.any(render_array[0, ..., 3] < 0.0):
                    raise GsVideoError("rasterizer expected depth must be nonnegative")
                return PickBuffer(
                    rgb=np.ascontiguousarray(self._rgb8(render_array)),
                    expected_depth=np.ascontiguousarray(
                        render_array[0, ..., 3], dtype=np.float32
                    ),
                    opacity=np.ascontiguousarray(alpha_array[0, ..., 0], dtype=np.float32),
                )
            finally:
                del render, alpha, meta
                prepared.metrics.release_frame()

    def _render_pick_locked(
        self, scene: GaussianScene, camera: CameraMatrices, width: int, height: int
    ) -> PickBuffer:
        settings = RenderSettings(
            width=width, height=height, sh_degree=self._scene_sh_degree(scene)
        )
        self._validate_scene_settings(scene, settings)
        rasterizer, metrics = self._dependencies()
        self._admit(scene, width, height, metrics)
        runtime = _runtime_scene(scene)
        render: object | None = None
        alpha: object | None = None
        meta: object | None = None
        try:
            render, alpha, meta = self._call(
                rasterizer, runtime, camera, settings, "RGB+ED"
            )
            render_array, alpha_array = self._validate_outputs(
                render, alpha, width=width, height=height, channels=4
            )
            if np.any(render_array[0, ..., 3] < 0.0):
                raise GsVideoError("rasterizer expected depth must be nonnegative")
            return PickBuffer(
                rgb=self._rgb8(render_array),
                expected_depth=np.ascontiguousarray(render_array[0, ..., 3], dtype=np.float32),
                opacity=np.ascontiguousarray(alpha_array[0, ..., 0], dtype=np.float32),
            )
        finally:
            del render, alpha, meta
            metrics.release_frame()
