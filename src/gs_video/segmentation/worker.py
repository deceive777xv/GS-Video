from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from gs_video.domain.errors import UnsupportedMaterialError

EventEmitter = Callable[[dict[str, object]], None]


def build_predictor(config: Path, checkpoint: Path) -> object:
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-not-found]

    return build_sam2_video_predictor(str(config), str(checkpoint))


def _frames(frames_dir: Path) -> list[Path]:
    candidates = [
        path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if any(not path.stem.isdigit() for path in candidates):
        raise ValueError("代理帧必须使用数字文件名")
    candidates.sort(key=lambda path: int(path.stem))
    if len({int(path.stem) for path in candidates}) != len(candidates):
        raise ValueError("代理帧数字编号不能重复")
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
) -> dict[str, object]:
    if backend not in {"edgetam", "sam2"}:
        raise ValueError("unknown segmentation backend")
    frames = _frames(frames_dir)
    if not frames:
        raise ValueError("未找到代理帧")
    if not 0 <= frame_index < len(frames):
        raise ValueError("提示帧索引越界")
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
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        return {"type": "result", "mask_dir": output_dir.name, "frames": len(frames)}
    finally:
        inference_stack.close()
        if staging.exists():
            shutil.rmtree(staging)
        del predictor


def _probe_builder() -> str:
    from sam2.build_sam import build_sam2_video_predictor

    if not callable(build_sam2_video_predictor):
        raise TypeError("build_sam2_video_predictor is not callable")
    module = build_sam2_video_predictor.__module__
    if not module.startswith("sam2."):
        raise ImportError("unexpected predictor builder module")
    return cast(str, module)


def probe_backend(
    backend: str,
    config: Path,
    checkpoint: Path,
    *,
    builder_probe: Callable[[], str] = _probe_builder,
) -> dict[str, object]:
    if backend not in {"edgetam", "sam2"}:
        raise ValueError("unknown segmentation backend")
    for path in (config, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            stream.read(1)
    builder = builder_probe()
    if not builder:
        raise ImportError("predictor builder identity is empty")
    return {
        "type": "probe",
        "backend": backend,
        "config": str(config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "builder": builder,
    }


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
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
