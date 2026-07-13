from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO, cast

from PIL import Image

from gs_video.domain.contracts import MaskSequence, Prompt, SegmentationBackend
from gs_video.domain.errors import GsVideoError, UnsupportedMaterialError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.segmentation.paths import has_reparse_component, worker_path
from gs_video.segmentation.tree_guard import (
    ProcessTreeGuard,
    create_process_tree_guard,
)

ProcessFactory = Callable[..., Any]
TreeGuardFactory = Callable[[Any], ProcessTreeGuard]


class VideoSegmenterClient:
    def __init__(
        self,
        *,
        backend: SegmentationBackend,
        worker_prefix: Sequence[str],
        model_config: Path,
        checkpoint: Path,
        process_factory: ProcessFactory = subprocess.Popen,
        tree_guard_factory: TreeGuardFactory = create_process_tree_guard,
        log_path: Path | None = None,
    ) -> None:
        if not worker_prefix or any(not item for item in worker_prefix):
            raise ValueError("worker prefix must contain nonempty argv entries")
        self.backend = backend
        self.worker_prefix = tuple(worker_prefix)
        self.model_config = Path(model_config)
        self.checkpoint = Path(checkpoint)
        self._process_factory = process_factory
        self._tree_guard_factory = tree_guard_factory
        self.log_path = log_path or self.model_config.parent / "logs" / "segmentation-worker.log"

    def _command(
        self, frames_dir: Path, output_dir: Path, prompt: Prompt, startup_gate: Path
    ) -> list[str]:
        return [
            *self.worker_prefix,
            "-m", "gs_video.segmentation.worker",
            "--backend", self.backend.value,
            "--frames", worker_path(frames_dir, self.worker_prefix),
            "--output", worker_path(output_dir, self.worker_prefix),
            "--frame-index", str(prompt.frame_index),
            "--point", f"{prompt.x},{prompt.y}",
            "--config", worker_path(self.model_config, self.worker_prefix),
            "--checkpoint", worker_path(self.checkpoint, self.worker_prefix),
            "--startup-gate", worker_path(startup_gate, self.worker_prefix),
        ]

    @staticmethod
    def _create_startup_gate(output_dir: Path) -> Path:
        parent = output_dir.parent.absolute()
        parent.mkdir(parents=True, exist_ok=True)
        if has_reparse_component(parent):
            raise GsVideoError("startup gate 父目录包含链接或重解析点")
        gate = parent / f".gs-video-segmentation-gate-{uuid.uuid4().hex}"
        with gate.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("WAIT\n")
        return gate

    @staticmethod
    def _release_startup_gate(gate: Path) -> None:
        release = gate.parent / f".{gate.name}.release-{uuid.uuid4().hex}"
        try:
            with release.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write("RELEASE\n")
            release.replace(gate)
        finally:
            release.unlink(missing_ok=True)

    @staticmethod
    def _remove_startup_gate(gate: Path) -> None:
        try:
            gate.unlink(missing_ok=True)
        except OSError:
            return

    @staticmethod
    def _reap_direct_best_effort(process: Any) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()
        except BaseException:
            pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def _start_worker(
        self, frames_dir: Path, output_dir: Path, prompt: Prompt
    ) -> tuple[Any, ProcessTreeGuard, Path]:
        options: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "shell": False,
        }
        if os.name == "nt":
            options["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            options["start_new_session"] = True
        gate = self._create_startup_gate(output_dir)
        try:
            process = self._process_factory(
                self._command(frames_dir, output_dir, prompt, gate), **options
            )
        except BaseException:
            self._remove_startup_gate(gate)
            raise
        try:
            guard = self._tree_guard_factory(process)
        except BaseException as exc:
            self._reap_direct_best_effort(process)
            self._remove_startup_gate(gate)
            raise GsVideoError("无法建立 segmentation worker 进程树隔离") from exc
        try:
            self._release_startup_gate(gate)
        except BaseException as exc:
            try:
                guard.close()
            except BaseException:
                pass
            self._reap_direct_best_effort(process)
            self._remove_startup_gate(gate)
            raise GsVideoError("无法释放 segmentation worker startup gate") from exc
        return process, guard, gate

    @staticmethod
    def _reader(stream: TextIO, target: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                target.put(line)
        except (OSError, ValueError):
            pass
        finally:
            target.put(None)

    @staticmethod
    def _stderr_reader(stream: TextIO, chunks: list[str]) -> None:
        try:
            chunks.append(stream.read())
        except (OSError, ValueError):
            return

    def _stop(self, process: Any, guard: ProcessTreeGuard) -> None:
        if process.poll() is None:
            if not guard.terminate(force=False):
                process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tree_killed = guard.terminate(force=True)
                if not tree_killed or process.poll() is None:
                    process.kill()
                process.wait()
        else:
            process.wait()

    def _write_stderr(self, chunks: list[str]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write("".join(chunks))

    @staticmethod
    def _is_link(path: Path) -> bool:
        return has_reparse_component(path)

    def _cleanup_process(
        self,
        process: Any,
        guard: ProcessTreeGuard,
        threads: tuple[threading.Thread | None, threading.Thread | None],
        stderr_chunks: list[str],
    ) -> None:
        truncated = False
        try:
            if process.poll() is None:
                self._stop(process, guard)
        except BaseException as exc:
            truncated = True
            stderr_chunks.append(f"\n[segmentation cleanup error: {exc}]\n")
        try:
            guard.close()
        except BaseException as exc:
            truncated = True
            stderr_chunks.append(f"\n[segmentation tree guard close error: {exc}]\n")
        live = [thread for thread in threads if thread is not None]
        for thread in live:
            thread.join(timeout=0.5)
        if any(thread.is_alive() for thread in live):
            truncated = True
            for thread in live:
                thread.join(timeout=0.1)
        else:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
        if any(thread.is_alive() for thread in live):
            truncated = True
        if truncated:
            stderr_chunks.append("\n[segmentation worker 日志截断：继承的管道句柄未及时关闭]\n")
        try:
            self._write_stderr(stderr_chunks)
        except OSError:
            return

    @staticmethod
    def _overlaps(first: Path, second: Path) -> bool:
        first_resolved = first.resolve()
        second_resolved = second.resolve()
        return (
            first_resolved == second_resolved
            or first_resolved in second_resolved.parents
            or second_resolved in first_resolved.parents
        )

    def _validate_request(
        self, frames: list[Path], prompt: Prompt, output_dir: Path
    ) -> list[Path]:
        if not frames:
            raise GsVideoError("分割至少一帧代理帧")
        for asset in (self.model_config, self.checkpoint):
            if self._is_link(asset) or not asset.is_file():
                raise GsVideoError("分割模型配置或 checkpoint 不可读")
            try:
                with asset.open("rb") as stream:
                    stream.read(1)
            except OSError as exc:
                raise GsVideoError("分割模型配置或 checkpoint 不可读") from exc
        allowed = {".jpg", ".jpeg", ".png"}
        if any(
            not path.stem.isdigit()
            or path.suffix.lower() not in allowed
            or self._is_link(path)
            or not path.is_file()
            for path in frames
        ):
            raise GsVideoError("代理帧必须是可读的非链接数字文件名 JPG/PNG")
        ordered = sorted(frames, key=lambda path: int(path.stem))
        if len({int(path.stem) for path in ordered}) != len(ordered):
            raise GsVideoError("代理帧数字编号不能重复")
        frames_dir = ordered[0].parent.resolve()
        if any(path.parent.resolve() != frames_dir for path in ordered):
            raise GsVideoError("代理帧必须位于同一目录")
        image_candidates = [path for path in frames_dir.iterdir() if path.suffix.lower() in allowed]
        if any(not path.stem.isdigit() for path in image_candidates):
            raise GsVideoError("代理帧目录包含非数字文件名图像")
        inventory = {path.resolve() for path in image_candidates}
        if {path.resolve() for path in ordered} != inventory:
            raise GsVideoError("代理帧必须等于完整目录数字图像清单")
        try:
            for path in ordered:
                with Image.open(path) as image:
                    image.verify()
        except (OSError, ValueError) as exc:
            raise GsVideoError("代理帧包含不可读图像") from exc
        if (
            type(prompt.frame_index) is not int
            or type(prompt.x) is not int
            or type(prompt.y) is not int
            or not 0 <= prompt.frame_index < len(ordered)
        ):
            raise GsVideoError("提示字段或帧索引无效")
        try:
            with Image.open(ordered[prompt.frame_index]) as image:
                width, height = image.size
                image.load()
        except (OSError, ValueError) as exc:
            raise GsVideoError("提示帧不可读") from exc
        if not 0 <= prompt.x < width or not 0 <= prompt.y < height:
            raise GsVideoError("提示点超出图像边界")
        if self._is_link(output_dir) or (output_dir.exists() and not output_dir.is_dir()):
            raise GsVideoError("输出目录必须是非链接目录")
        if self._overlaps(output_dir, frames_dir) or any(
            self._overlaps(output_dir, asset) for asset in (self.model_config, self.checkpoint)
        ):
            raise GsVideoError("输出目录与输入或模型路径重叠")
        return ordered

    def _wait_for_exit(self, process: Any, token: CancellationToken) -> int:
        while True:
            try:
                return cast(int, process.wait(timeout=0.05))
            except subprocess.TimeoutExpired:
                token.raise_if_cancelled()

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
        event: dict[str, object], output_dir: Path, expected_sizes: dict[str, tuple[int, int]]
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
        if frames_value != len(expected_sizes):
            raise GsVideoError("segmentation worker mask 数量不匹配")
        if not output_dir.is_dir() or VideoSegmenterClient._is_link(output_dir):
            raise GsVideoError("segmentation worker mask 输出目录无效")
        entries = list(output_dir.iterdir())
        if any(
            path.suffix.lower() != ".png"
            or VideoSegmenterClient._is_link(path)
            or not path.is_file()
            for path in entries
        ):
            raise GsVideoError("segmentation worker mask 文件无效")
        actual_names = tuple(sorted(path.name for path in entries))
        expected_names = tuple(sorted(expected_sizes))
        if actual_names != expected_names:
            raise GsVideoError("segmentation worker mask 文件缺失或命名错误")
        try:
            for path in entries:
                with Image.open(path) as image:
                    image.load()
                    if (
                        image.format != "PNG"
                        or image.mode != "L"
                        or image.size != expected_sizes[path.name]
                    ):
                        raise GsVideoError("segmentation worker mask PNG 格式、模式或尺寸无效")
        except (OSError, ValueError) as exc:
            raise GsVideoError("segmentation worker mask PNG 不可读") from exc
        return MaskSequence(mask_dir=output_dir, frame_count=frames_value)

    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence:
        ordered = self._validate_request(frames, prompt, output_dir)
        frames_dir = ordered[0].parent
        try:
            process, guard, startup_gate = self._start_worker(frames_dir, output_dir, prompt)
        except ValueError as exc:
            raise GsVideoError("segmentation worker 路径无效") from exc
        lines: queue.Queue[str | None] = queue.Queue()
        stderr_chunks: list[str] = []
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        terminal: dict[str, object] | None = None
        last_current = 0
        exit_drain_deadline: float | None = None
        try:
            if process.stdout is None or process.stderr is None:
                raise GsVideoError("segmentation worker 管道不可用")
            stdout_thread = threading.Thread(
                target=self._reader, args=(process.stdout, lines), daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._stderr_reader, args=(process.stderr, stderr_chunks), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            while True:
                token.raise_if_cancelled()
                if process.poll() is not None and exit_drain_deadline is None:
                    exit_drain_deadline = time.monotonic() + 1.0
                timeout = 0.05
                if exit_drain_deadline is not None:
                    remaining = exit_drain_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    timeout = min(timeout, remaining)
                try:
                    line = lines.get(timeout=timeout)
                except queue.Empty:
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
                        or total != len(ordered)
                        or current != last_current + 1
                        or current > total
                    ):
                        raise GsVideoError("segmentation worker progress 非单调或无效")
                    last_current = current
                    emit(current, total, f"分割前景 {current}/{total}")
                else:
                    if terminal is not None:
                        raise GsVideoError("segmentation worker 返回多个终止事件")
                    terminal = event
            returncode = self._wait_for_exit(process, token)
            if terminal is None:
                if returncode != 0:
                    raise GsVideoError(f"segmentation worker 异常退出: {returncode}")
                raise GsVideoError("segmentation worker 未返回终止事件")
            if terminal["type"] == "error":
                code, message = terminal["code"], terminal["message"]
                if not isinstance(code, str) or not isinstance(message, str):
                    raise GsVideoError("segmentation worker error schema 无效")
                if code == "unsupported_material":
                    if returncode != 0:
                        raise GsVideoError("segmentation worker error/exit 状态不匹配")
                    raise UnsupportedMaterialError(message)
                if code == "system_error" and returncode != 0:
                    raise GsVideoError(message)
                raise GsVideoError("segmentation worker error/exit 状态或 code 无效")
            if returncode != 0:
                raise GsVideoError("segmentation worker result/exit 状态不匹配")
            if last_current != len(ordered):
                raise GsVideoError("segmentation worker result 前 progress 未完成")
            expected_sizes: dict[str, tuple[int, int]] = {}
            for path in ordered:
                with Image.open(path) as image:
                    expected_sizes[f"{path.stem}.png"] = image.size
            return self._validate_result(terminal, output_dir, expected_sizes)
        finally:
            if stderr_thread is None and process.stderr is not None:
                try:
                    stderr_chunks.append(process.stderr.read())
                except (OSError, ValueError):
                    pass
            self._cleanup_process(process, guard, (stdout_thread, stderr_thread), stderr_chunks)
            self._remove_startup_gate(startup_gate)
