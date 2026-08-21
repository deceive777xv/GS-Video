from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from gs_video.camera.serialization import read_camera_solution
from gs_video.camera.solution import CameraSolution
from gs_video.domain.errors import GsVideoError, RepairableError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.pipeline.gpu import GpuAdmissionGate
from gs_video.segmentation.paths import (
    has_reparse_component,
    is_wsl_prefix,
    worker_path,
)
from gs_video.segmentation.tree_guard import create_process_tree_guard


ProcessFactory = Callable[..., Any]


class VipeCameraSolver:
    """Run the pinned ViPE backend in its isolated Python environment."""

    identity = "nvidia-vipe-1.2.0-raw-video-v1"

    def __init__(
        self,
        worker_prefix: Sequence[str],
        *,
        process_factory: ProcessFactory = subprocess.Popen,
        gpu_gate: GpuAdmissionGate | None = None,
        cache_root: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        if not worker_prefix or any(not value for value in worker_prefix):
            raise ValueError("ViPE worker prefix must contain nonempty argv entries")
        self.worker_prefix = tuple(worker_prefix)
        self._process_factory = process_factory
        self._gpu_gate = gpu_gate
        self._cache_root = None if cache_root is None else Path(cache_root).absolute()
        self._log_path = log_path

    @staticmethod
    def _reader(stream: TextIO, target: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                target.put(line)
        finally:
            target.put(None)

    @staticmethod
    def _stderr_reader(stream: TextIO, chunks: list[str]) -> None:
        try:
            chunks.append(stream.read())
        except (OSError, ValueError):
            return

    @staticmethod
    def _validate_inventory(
        frames: Sequence[Path], masks: Sequence[Path], output_dir: Path
    ) -> tuple[Path, Path]:
        if not frames or len(frames) != len(masks):
            raise ValueError("ViPE requires one subject mask per source frame")
        frame_dir = Path(frames[0]).parent.absolute()
        mask_dir = Path(masks[0]).parent.absolute()
        if any(Path(path).parent.absolute() != frame_dir for path in frames):
            raise ValueError("ViPE frames must share one directory")
        if any(Path(path).parent.absolute() != mask_dir for path in masks):
            raise ValueError("ViPE masks must share one directory")
        if any(
            has_reparse_component(Path(path)) or not Path(path).is_file()
            for path in (*frames, *masks)
        ):
            raise ValueError("ViPE frame and mask inputs must be ordinary files")
        if [Path(path).stem for path in frames] != [Path(path).stem for path in masks]:
            raise ValueError("ViPE frame and mask indices must match")
        output = Path(output_dir).absolute()
        if has_reparse_component(output) or not output.is_dir():
            raise ValueError("ViPE output must be an existing ordinary directory")
        if output in frame_dir.parents or output in mask_dir.parents:
            raise ValueError("ViPE output must not contain its inputs")
        return frame_dir, mask_dir

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        if self._cache_root is not None:
            self._cache_root.mkdir(parents=True, exist_ok=True)
            cache_variables = {
                "HF_HOME": str(self._cache_root / "huggingface"),
                "HF_HUB_CACHE": str(self._cache_root / "huggingface" / "hub"),
                "TORCH_HOME": str(self._cache_root / "torch"),
                "XDG_CACHE_HOME": str(self._cache_root),
            }
            environment.update(cache_variables)
            if is_wsl_prefix(self.worker_prefix):
                existing = [
                    item
                    for item in environment.get("WSLENV", "").split(":")
                    if item
                ]
                known = {item.split("/", 1)[0] for item in existing}
                existing.extend(
                    f"{name}/p" for name in cache_variables if name not in known
                )
                environment["WSLENV"] = ":".join(existing)
        return environment

    def _command(
        self,
        frame_dir: Path,
        mask_dir: Path,
        output_dir: Path,
        count: int,
    ) -> list[str]:
        return [
            *self.worker_prefix,
            "-m",
            "gs_video.camera.vipe_worker",
            "--frames",
            worker_path(frame_dir, self.worker_prefix),
            "--masks",
            worker_path(mask_dir, self.worker_prefix),
            "--output",
            worker_path(output_dir, self.worker_prefix),
            "--count",
            str(count),
        ]

    def solve(
        self,
        frame_paths: list[Path] | tuple[Path, ...],
        mask_paths: list[Path] | tuple[Path, ...],
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> CameraSolution:
        frames = tuple(Path(path).absolute() for path in frame_paths)
        masks = tuple(Path(path).absolute() for path in mask_paths)
        frame_dir, mask_dir = self._validate_inventory(frames, masks, output_dir)
        command = self._command(
            frame_dir,
            mask_dir,
            Path(output_dir).absolute(),
            len(frames),
        )
        options: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "shell": False,
            "env": self._environment(),
        }
        if os.name == "nt":
            options["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        gate = self._gpu_gate.hold(token) if self._gpu_gate is not None else None
        if gate is not None:
            gate.__enter__()
        process: Any | None = None
        guard: Any | None = None
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        stderr_chunks: list[str] = []
        messages: queue.Queue[str | None] = queue.Queue()
        complete = False
        try:
            token.raise_if_cancelled()
            process = self._process_factory(command, **options)
            guard = create_process_tree_guard(process)
            assert process.stdout is not None and process.stderr is not None
            stdout_thread = threading.Thread(
                target=self._reader, args=(process.stdout, messages), daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._stderr_reader,
                args=(process.stderr, stderr_chunks),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            while True:
                token.raise_if_cancelled()
                try:
                    line = messages.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None:
                        break
                    continue
                if line is None:
                    break
                if len(line.encode("utf-8")) > 64 * 1024:
                    raise GsVideoError("ViPE worker event exceeds the size limit")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise GsVideoError("ViPE worker emitted invalid JSON") from error
                if event.get("type") == "progress":
                    emit(int(event["current"]), int(event["total"]), str(event["message"]))
                elif event.get("type") == "complete":
                    complete = True
                elif event.get("type") == "error":
                    raise RepairableError(str(event.get("message", "ViPE camera solve failed")))
                else:
                    raise GsVideoError("ViPE worker emitted an unknown event")
            return_code = process.wait()
            if return_code != 0 or not complete:
                raise RepairableError("ViPE camera solve failed; no fallback was attempted")
            solution = read_camera_solution(Path(output_dir) / "solution.json")
            if solution.source_ground is None or not (Path(output_dir) / "depth.zip").is_file():
                raise RepairableError("ViPE output lacks audited depth or source ground")
            return solution
        finally:
            if process is not None and process.poll() is None:
                if guard is not None:
                    guard.terminate(force=False)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if guard is not None:
                        guard.terminate(force=True)
                    process.kill()
                    process.wait()
            if guard is not None:
                guard.close()
            for thread in (stdout_thread, stderr_thread):
                if thread is not None:
                    thread.join(timeout=1)
            if self._log_path is not None and stderr_chunks:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as stream:
                    stream.write("".join(stderr_chunks))
            if gate is not None:
                gate.__exit__(None, None, None)
