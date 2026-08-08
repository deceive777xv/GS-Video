from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from gs_video.domain.models import SceneSummary
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.preview_session import PreviewSession


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the persistent Gaussian preview session on an imported scene."
    )
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.frames < 2:
        parser.error("--frames must be at least 2")

    config = json.loads(arguments.runtime_config.read_text(encoding="utf-8"))
    project_root = Path(config["project_root"]).resolve(strict=True)
    project = json.loads((project_root / "project.json").read_text(encoding="utf-8"))
    summary = SceneSummary.model_validate(project["workflow"]["scene_summary"])
    selected = project["workflow"]["target_camera"]
    camera = OrbitCamera(
        target=tuple(selected["target"]),
        distance=selected["distance"],
        yaw=selected["yaw"],
        pitch=selected["pitch"],
        fov_y_degrees=selected["fov_y_degrees"],
    )
    session = PreviewSession(
        worker_prefix=tuple(config["renderer_worker_prefix"]),
        sh_degree=int(config["renderer_sh_degree"]),
        available_vram_limit_mb=int(config["available_vram_limit_mb"]),
        log_path=project_root / "logs" / "preview-session-benchmark.log",
    )
    close_seconds = 0.0
    try:
        started = perf_counter()
        first = session.render_live(
            project_root, project["scene_ply"], summary, 1, camera, 960, 540
        )
        setup_seconds = perf_counter() - started
        warm_seconds: list[float] = []
        gpu_raster_ms: list[float] = []
        readback_ms: list[float] = []
        jpeg_ms: list[float] = []
        sizes = [len(first)]
        for index in range(2, arguments.frames + 1):
            moved = OrbitCamera(
                target=camera.target,
                distance=camera.distance,
                yaw=camera.yaw + index * 0.2,
                pitch=camera.pitch,
                fov_y_degrees=camera.fov_y_degrees,
            )
            started = perf_counter()
            payload = session.render_live(
                project_root,
                project["scene_ply"],
                summary,
                index,
                moved,
                960,
                540,
            )
            warm_seconds.append(perf_counter() - started)
            timings = session.last_live_timings
            if timings is None:
                raise RuntimeError("preview worker did not report segmented timings")
            gpu_raster_ms.append(timings.gpu_raster_ms)
            readback_ms.append(timings.readback_ms)
            jpeg_ms.append(timings.jpeg_ms)
            sizes.append(len(payload))
        started = perf_counter()
        pick = session.render_preview_pick(
            project_root, project["scene_ply"], summary, camera, 960, 540
        )
        authoritative_seconds = perf_counter() - started
    finally:
        started = perf_counter()
        session.close()
        close_seconds = perf_counter() - started

    print(
        json.dumps(
            {
                "gaussian_count": summary.gaussian_count,
                "scene_size_bytes": summary.size,
                "session_setup_and_first_live_ms": round(setup_seconds * 1000, 2),
                "warm_live_mean_ms": round(mean(warm_seconds) * 1000, 2),
                "warm_live_median_ms": round(median(warm_seconds) * 1000, 2),
                "warm_live_p95_ms": round(_percentile(warm_seconds, 0.95) * 1000, 2),
                "warm_live_fps": round(1 / mean(warm_seconds), 2),
                "warm_gpu_raster_mean_ms": round(mean(gpu_raster_ms), 2),
                "warm_readback_mean_ms": round(mean(readback_ms), 2),
                "warm_jpeg_mean_ms": round(mean(jpeg_ms), 2),
                "jpeg_mean_kib": round(mean(sizes) / 1024, 1),
                "authoritative_rgb_depth_ms": round(authoritative_seconds * 1000, 2),
                "authoritative_rgb_shape": list(pick.rgb.shape),
                "authoritative_depth_shape": list(pick.expected_depth.shape),
                "close_ms": round(close_seconds * 1000, 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
