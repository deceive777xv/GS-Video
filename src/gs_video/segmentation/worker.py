from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from gs_video.domain.errors import UnsupportedMaterialError

EventEmitter = Callable[[dict[str, object]], None]
PathReplacer = Callable[[Path, Path], None]


def build_predictor(config: Path, checkpoint: Path) -> object:
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-not-found]

    return build_sam2_video_predictor(str(config), str(checkpoint))


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _readable_regular_file(path: Path) -> bool:
    if _is_link(path) or not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except OSError:
        return False
    return True


def _frames(frames_dir: Path) -> list[Path]:
    if _is_link(frames_dir) or not frames_dir.is_dir():
        raise ValueError("代理帧目录必须是非链接目录")
    candidates = [
        path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if any(not path.stem.isdigit() for path in candidates):
        raise ValueError("代理帧必须使用数字文件名")
    if any(not _readable_regular_file(path) for path in candidates):
        raise ValueError("代理帧必须是可读的非链接普通文件")
    candidates.sort(key=lambda path: int(path.stem))
    if len({int(path.stem) for path in candidates}) != len(candidates):
        raise ValueError("代理帧数字编号不能重复")
    try:
        for path in candidates:
            with Image.open(path) as image:
                image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError("代理帧包含不可读图像") from exc
    return candidates


def _numpy_mask(logits: object) -> np.ndarray:
    value: Any = cast(Any, logits)[0]
    value = value > 0
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value).squeeze()
    if array.ndim != 2:
        raise ValueError("predictor mask 必须是二维图像")
    return array.astype(np.uint8) * 255


def _has_invisible_run(ratios: list[float]) -> bool:
    run = 0
    for ratio in ratios:
        run = run + 1 if ratio < 0.001 else 0
        if run >= 15:
            return True
    return False


def _overlaps(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _replace(source: Path, destination: Path) -> None:
    source.replace(destination)


def _promote_staging(
    staging: Path,
    output_dir: Path,
    replace_path: PathReplacer = _replace,
) -> None:
    backup: Path | None = None
    if _is_link(output_dir):
        raise ValueError("输出路径必须是非链接目录")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("输出路径必须是非链接目录")
        backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        replace_path(output_dir, backup)
    try:
        replace_path(staging, output_dir)
    except BaseException:
        if backup is not None and backup.exists() and not output_dir.exists():
            replace_path(backup, output_dir)
        raise
    if backup is not None and backup.exists():
        if _is_link(backup) or not backup.is_dir() or backup.parent != output_dir.parent:
            raise RuntimeError("拒绝清理不安全的输出备份")
        shutil.rmtree(backup)


def run_segmentation(
    *,
    backend: str,
    frames_dir: Path,
    output_dir: Path,
    frame_index: int,
    point: tuple[int, int],
    config: Path,
    checkpoint: Path,
    predictor_factory: Callable[[Path, Path], object] = build_predictor,
    emit: EventEmitter,
    replace_path: PathReplacer = _replace,
) -> dict[str, object]:
    if backend not in {"edgetam", "sam2"}:
        raise ValueError("unknown segmentation backend")
    frames = _frames(frames_dir)
    if not frames:
        raise ValueError("未找到代理帧")
    if type(frame_index) is not int or not 0 <= frame_index < len(frames):
        raise ValueError("提示帧索引越界")
    if (
        type(point[0]) is not int
        or type(point[1]) is not int
        or point[0] < 0
        or point[1] < 0
    ):
        raise ValueError("提示点无效")
    try:
        with Image.open(frames[frame_index]) as image:
            width, height = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError("提示帧不可读") from exc
    if point[0] >= width or point[1] >= height:
        raise ValueError("提示点超出图像边界")
    for asset in (config, checkpoint):
        if not _readable_regular_file(asset):
            raise ValueError("模型配置或 checkpoint 不可读")
    if _is_link(output_dir) or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError("输出路径必须是非链接目录")
    if _overlaps(output_dir, frames_dir) or any(
        _overlaps(output_dir, asset) for asset in (config, checkpoint)
    ):
        raise ValueError("输出路径与输入或模型路径重叠")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    predictor: object | None = None
    inference_stack = contextlib.ExitStack()
    try:
        try:
            import torch  # type: ignore[import-not-found]

            inference_stack.enter_context(torch.inference_mode())
            if torch.cuda.is_available():
                inference_stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        except ImportError:
            pass
        predictor = predictor_factory(config, checkpoint)
        state = predictor.init_state(video_path=str(frames_dir))  # type: ignore[attr-defined]
        predictor.add_new_points_or_box(  # type: ignore[attr-defined]
            state,
            frame_idx=frame_index,
            obj_id=1,
            points=np.array([[float(point[0]), float(point[1])]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        written: set[int] = set()
        ratios_by_index: dict[int, float] = {}
        for reverse in (False, True):
            propagation = predictor.propagate_in_video(  # type: ignore[attr-defined]
                state, start_frame_idx=frame_index, reverse=reverse
            )
            for index, _object_ids, logits in propagation:
                if index in written:
                    continue
                if not isinstance(index, int) or not 0 <= index < len(frames):
                    raise ValueError("predictor 返回无效帧索引")
                mask = _numpy_mask(logits)
                Image.fromarray(mask).save(staging / f"{frames[index].stem}.png")
                written.add(index)
                ratios_by_index[index] = float(np.count_nonzero(mask)) / float(mask.size)
                emit({"type": "progress", "current": len(written), "total": len(frames)})
        if written != set(range(len(frames))):
            raise RuntimeError("predictor 未覆盖全部代理帧")
        if _has_invisible_run([ratios_by_index[index] for index in range(len(frames))]):
            raise UnsupportedMaterialError("主要人物长时间不可见")
        _promote_staging(staging, output_dir, replace_path)
        return {"type": "result", "mask_dir": output_dir.name, "frames": len(frames)}
    finally:
        inference_stack.close()
        if staging.exists():
            shutil.rmtree(staging)
        del predictor


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def probe_backend(
    backend: str,
    config: Path,
    checkpoint: Path,
    *,
    predictor_factory: Callable[[Path, Path], object] = build_predictor,
    cuda_cleanup: Callable[[], None] = _clear_cuda_cache,
) -> dict[str, object]:
    if backend not in {"edgetam", "sam2"}:
        raise ValueError("unknown segmentation backend")
    for path in (config, checkpoint):
        if not _readable_regular_file(path):
            raise FileNotFoundError(path)
    predictor: object | None = None
    try:
        predictor = predictor_factory(config, checkpoint)
        identity = f"{type(predictor).__module__}.{type(predictor).__qualname__}"
        if predictor is None or not identity:
            raise RuntimeError("predictor build returned invalid object")
        return {
            "type": "probe",
            "backend": backend,
            "config": str(config.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "predictor": identity,
        }
    finally:
        del predictor
        cuda_cleanup()


def _point(value: str) -> tuple[int, int]:
    try:
        x, y = value.split(",", maxsplit=1)
        return int(x), int(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point must be x,y") from exc


def _emit(event: dict[str, object]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("edgetam", "sam2"), required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--point", type=_point)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    protocol_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.probe:
                result = probe_backend(args.backend, args.config, args.checkpoint)
            else:
                if None in (args.frames, args.output, args.frame_index, args.point):
                    parser.error("segmentation arguments are required unless --probe is used")
                result = run_segmentation(
                    backend=args.backend,
                    frames_dir=args.frames,
                    output_dir=args.output,
                    frame_index=args.frame_index,
                    point=args.point,
                    config=args.config,
                    checkpoint=args.checkpoint,
                    emit=lambda event: print(
                        json.dumps(event, ensure_ascii=False), file=protocol_stdout, flush=True
                    ),
                )
        print(json.dumps(result, ensure_ascii=False), file=protocol_stdout, flush=True)
        return 0
    except UnsupportedMaterialError as exc:
        _emit({"type": "error", "code": exc.code, "message": str(exc)})
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        _emit({"type": "error", "code": "system_error", "message": "分割 worker 失败"})
        return 1
    finally:
        _clear_cuda_cache()


if __name__ == "__main__":
    raise SystemExit(main())
