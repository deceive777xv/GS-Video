from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gs_video.postprocess.lut import parse_cube
from gs_video.postprocess.worker_protocol import read_request


IMPLEMENTATION_VERSION = "cuda-post-process-worker-v1"
MAX_SESSION_COMMAND_BYTES = 32 * 1024


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def _read_rgb16(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("input is not a 16-bit RGB PNG")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def _write_rgb16(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError("could not write a processed 16-bit RGB PNG")


def _configure_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for post-processing")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _process(request_path: Path) -> None:
    from gs_video.postprocess.torch_engine import process_rgb16_cuda

    request = read_request(request_path)
    luts = {
        asset_id: parse_cube(path.read_text(encoding="utf-8-sig"))
        for asset_id, path in request.lut_paths.items()
    }
    total = len(request.frame_paths)
    for index, frame_path in enumerate(request.frame_paths, start=1):
        source = _read_rgb16(frame_path)
        processed = process_rgb16_cuda(
            source,
            request.effects,
            resolve_lut=luts.__getitem__,
            spatial_scale=request.spatial_scale,
            vram_limit_mb=request.vram_limit_mb,
        )
        output = request.output_dir / f"{index:06d}.png"
        _write_rgb16(output, processed)
        _emit(
            {
                "type": "progress",
                "current": index,
                "total": total,
                "message": f"CUDA 后处理帧 {index}/{total}",
            }
        )
    _emit(
        {
            "type": "complete",
            "implementation_version": IMPLEMENTATION_VERSION,
            "outputs": [f"{index:06d}.png" for index in range(1, total + 1)],
        }
    )


def _emit_error(error: BaseException) -> None:
    message = str(error) or type(error).__name__
    code = (
        "postprocess_cuda_unavailable"
        if "CUDA is unavailable" in message
        else "postprocess_vram_exhausted"
        if "VRAM" in message or "out of memory" in message.lower()
        else "postprocess_worker_failed"
    )
    _emit({"type": "error", "code": code, "message": message[:512]})


def run(request_path: Path) -> int:
    try:
        _configure_cuda()
        _process(request_path)
        return 0
    except BaseException as error:
        _emit_error(error)
        return 1


def run_session() -> int:
    try:
        _configure_cuda()
        _emit({"type": "ready", "implementation_version": IMPLEMENTATION_VERSION})
        while True:
            line = sys.stdin.buffer.readline(MAX_SESSION_COMMAND_BYTES + 1)
            if line == b"":
                return 0
            if len(line) > MAX_SESSION_COMMAND_BYTES or not line.endswith(b"\n"):
                raise ValueError("post-process session command is invalid")
            command = json.loads(line.decode("utf-8"))
            if not isinstance(command, dict) or set(command) != {"request"}:
                raise ValueError("post-process session command is invalid")
            request_path = command["request"]
            if not isinstance(request_path, str) or not request_path:
                raise ValueError("post-process session request path is invalid")
            _process(Path(request_path))
    except BaseException as error:
        _emit_error(error)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--session", action="store_true")
    arguments = parser.parse_args()
    if arguments.session:
        return run_session()
    assert arguments.request is not None
    return run(arguments.request)


if __name__ == "__main__":
    raise SystemExit(main())
