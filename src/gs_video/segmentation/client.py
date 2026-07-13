from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO, cast

from gs_video.domain.contracts import MaskSequence, Prompt, SegmentationBackend
from gs_video.domain.errors import CancelledError, GsVideoError, UnsupportedMaterialError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter

ProcessFactory = Callable[..., Any]


class VideoSegmenterClient:
    def __init__(
        self,
        *,
        backend: SegmentationBackend,
        worker_prefix: Sequence[str],
        model_config: Path,
        checkpoint: Path,
        process_factory: ProcessFactory = subprocess.Popen,
        log_path: Path | None = None,
    ) -> None:
        if not worker_prefix or any(not item for item in worker_prefix):
            raise ValueError("worker prefix must contain nonempty argv entries")
        self.backend = backend
        self.worker_prefix = tuple(worker_prefix)
        self.model_config = Path(model_config)
        self.checkpoint = Path(checkpoint)
        self._process_factory = process_factory
        self.log_path = log_path or self.model_config.parent / "logs" / "segmentation-worker.log"

    def _command(self, frames_dir: Path, output_dir: Path, prompt: Prompt) -> list[str]:
        return [
            *self.worker_prefix,
            "-m", "gs_video.segmentation.worker",
            "--backend", self.backend.value,
            "--frames", str(frames_dir),
            "--output", str(output_dir),
            "--frame-index", str(prompt.frame_index),
            "--point", f"{prompt.x},{prompt.y}",
            "--config", str(self.model_config),
            "--checkpoint", str(self.checkpoint),
        ]

    def _start_worker(self, frames_dir: Path, output_dir: Path, prompt: Prompt) -> Any:
        options: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "shell": False,
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        return self._process_factory(self._command(frames_dir, output_dir, prompt), **options)

    @staticmethod
    def _reader(stream: TextIO, target: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                target.put(line)
        finally:
            target.put(None)

    @staticmethod
    def _stderr_reader(stream: TextIO, chunks: list[str]) -> None:
        chunks.append(stream.read())

    def _stop(self, process: Any) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.wait()

    def _write_stderr(self, chunks: list[str]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write("".join(chunks))

    def _wait_for_exit(self, process: Any, token: CancellationToken) -> int:
        while True:
            try:
                return cast(int, process.wait(timeout=0.05))
            except subprocess.TimeoutExpired:
                try:
                    token.raise_if_cancelled()
                except CancelledError:
                    self._stop(process)
                    raise

    @staticmethod
    def _parse_event(line: str) -> dict[str, object]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GsVideoError("segmentation worker 返回无效 JSONL") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
            raise GsVideoError("segmentation worker 事件格式无效")
        schemas = {
            "progress": {"type", "current", "total"},
            "result": {"type", "mask_dir", "frames"},
            "error": {"type", "code", "message"},
        }
        event_type = payload["type"]
        if event_type not in schemas or set(payload) != schemas[event_type]:
            raise GsVideoError("segmentation worker 事件 schema 无效")
        return payload

    @staticmethod
    def _validate_result(
        event: dict[str, object], output_dir: Path, expected_names: tuple[str, ...]
    ) -> MaskSequence:
        mask_dir_value = event["mask_dir"]
        frames_value = event["frames"]
        if (
            not isinstance(mask_dir_value, str)
            or not isinstance(frames_value, int)
            or isinstance(frames_value, bool)
        ):
            raise GsVideoError("segmentation worker result schema 无效")
        advertised = Path(mask_dir_value)
        actual = advertised if advertised.is_absolute() else output_dir.parent / advertised
        if actual.resolve() != output_dir.resolve():
            raise GsVideoError("segmentation worker result 路径越界")
        if frames_value != len(expected_names):
            raise GsVideoError("segmentation worker mask 数量不匹配")
        actual_names = tuple(sorted(path.name for path in output_dir.glob("*.png")))
        if actual_names != expected_names:
            raise GsVideoError("segmentation worker mask 文件缺失或命名错误")
        return MaskSequence(mask_dir=output_dir, frame_count=frames_value)

    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence:
        if any(not path.stem.isdigit() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"} for path in frames):
            raise GsVideoError("代理帧必须使用数字文件名的 JPG/PNG")
        ordered = sorted(frames, key=lambda path: int(path.stem))
        if len({int(path.stem) for path in ordered}) != len(ordered):
            raise GsVideoError("代理帧数字编号不能重复")
        if ordered and any(path.parent.resolve() != ordered[0].parent.resolve() for path in ordered):
            raise GsVideoError("代理帧必须位于同一目录")
        frames_dir = ordered[0].parent if ordered else output_dir.parent
        process = self._start_worker(frames_dir, output_dir, prompt)
        if process.stdout is None or process.stderr is None:
            raise GsVideoError("segmentation worker 管道不可用")
        lines: queue.Queue[str | None] = queue.Queue()
        stderr_chunks: list[str] = []
        stdout_thread = threading.Thread(target=self._reader, args=(process.stdout, lines), daemon=True)
        stderr_thread = threading.Thread(
            target=self._stderr_reader, args=(process.stderr, stderr_chunks), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        terminal: dict[str, object] | None = None
        last_current = 0
        try:
            while True:
                try:
                    token.raise_if_cancelled()
                except CancelledError:
                    self._stop(process)
                    raise
                try:
                    line = lines.get(timeout=0.05)
                except queue.Empty:
                    if process.poll() is not None and not stdout_thread.is_alive():
                        break
                    continue
                if line is None:
                    break
                event = self._parse_event(line)
                kind = event["type"]
                if kind == "progress":
                    current, total = event["current"], event["total"]
                    if (
                        terminal is not None
                        or not isinstance(current, int)
                        or isinstance(current, bool)
                        or not isinstance(total, int)
                        or isinstance(total, bool)
                        or total < 0 or current <= last_current or current > total
                    ):
                        raise GsVideoError("segmentation worker progress 非单调或无效")
                    last_current = current
                    emit(current, total, f"分割前景 {current}/{total}")
                else:
                    if terminal is not None:
                        raise GsVideoError("segmentation worker 返回多个终止事件")
                    terminal = event
            returncode = self._wait_for_exit(process, token)
            if returncode != 0:
                raise GsVideoError(f"segmentation worker 异常退出: {returncode}")
            if terminal is None:
                raise GsVideoError("segmentation worker 未返回终止事件")
            if terminal["type"] == "error":
                code, message = terminal["code"], terminal["message"]
                if not isinstance(code, str) or not isinstance(message, str):
                    raise GsVideoError("segmentation worker error schema 无效")
                if code == "unsupported_material":
                    raise UnsupportedMaterialError(message)
                raise GsVideoError(message)
            expected = tuple(sorted(f"{path.stem}.png" for path in ordered))
            return self._validate_result(terminal, output_dir, expected)
        finally:
            if process.poll() is None:
                self._stop(process)
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            self._write_stderr(stderr_chunks)
