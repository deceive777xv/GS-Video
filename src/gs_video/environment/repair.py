from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, RLock, Thread, current_thread
import time
from typing import Any
from uuid import uuid4

from gs_video.api.schemas import (
    EnvironmentRepairError,
    EnvironmentRepairSnapshot,
    EnvironmentRepairState,
)
from gs_video.environment.doctor import EnvironmentReport


class EnvironmentRepairBusyError(RuntimeError):
    """Raised when another repair runner owns the runtime lease."""


class EnvironmentRepairManager:
    """Owns one bounded environment-repair subprocess for an API instance."""

    _LEASE_STALE_SECONDS = 15.0
    _CANCEL_WAIT_SECONDS = 5.0

    def __init__(
        self,
        *,
        repo_root: Path,
        runtime_root: Path,
        runtime_config: Path,
        environment_doctor: Any,
        runner_path: Path | None = None,
        manifest_path: Path | None = None,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self._repo_root = repo_root.resolve(strict=False)
        self._runtime_root = runtime_root.resolve(strict=False)
        self._runtime_config = runtime_config.resolve(strict=False)
        self._environment_doctor = environment_doctor
        self._runner_path = (runner_path or self._repo_root / "tools" / "environment_repair.py").resolve(
            strict=False
        )
        self._manifest_path = (
            manifest_path or self._repo_root / "tools" / "runtime-manifest.json"
        ).resolve(strict=False)
        self._popen = popen
        self._lock = RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader: Thread | None = None
        self._heartbeat: Thread | None = None
        self._heartbeat_stop = Event()
        self._cancel_file: Path | None = None
        self._lease_path = self._runtime_root / "repair" / "lease.json"
        self._snapshot = EnvironmentRepairSnapshot(state=EnvironmentRepairState.IDLE)

    @property
    def runtime_root(self) -> Path:
        return self._runtime_root

    def snapshot(self) -> EnvironmentRepairSnapshot:
        with self._lock:
            current = self._snapshot.model_copy(deep=True)
        if current.state in {
            EnvironmentRepairState.IDLE,
            EnvironmentRepairState.CANCELLED,
            EnvironmentRepairState.FAILED,
        }:
            return current.model_copy(update={"resume_available": self._resume_available()})
        return current

    def is_busy(self) -> bool:
        with self._lock:
            return self._snapshot.state in {
                EnvironmentRepairState.RUNNING,
                EnvironmentRepairState.CANCELLING,
            }

    def start(self) -> EnvironmentRepairSnapshot:
        with self._lock:
            if self._snapshot.state in {
                EnvironmentRepairState.RUNNING,
                EnvironmentRepairState.CANCELLING,
            }:
                return self._snapshot.model_copy(deep=True)
            self._runtime_root.mkdir(parents=True, exist_ok=True)
            repair_root = self._runtime_root / "repair"
            repair_root.mkdir(parents=True, exist_ok=True)
            job_id = f"repair-{uuid4().hex}"
            self._claim_lease(repair_root / "lease.json", job_id)
            cancel_file = repair_root / f"{job_id}.cancel"
            cancel_file.unlink(missing_ok=True)
            self._cancel_file = cancel_file
            self._snapshot = EnvironmentRepairSnapshot(
                state=EnvironmentRepairState.RUNNING,
                job_id=job_id,
                step="preflight",
                progress=0.0,
                message="正在准备环境修复…",
                resume_available=self._resume_available(),
            )
            command = [
                sys.executable,
                str(self._runner_path),
                "--repo-root",
                str(self._repo_root),
                "--runtime-root",
                str(self._runtime_root),
                "--runtime-config",
                str(self._runtime_config),
                "--manifest",
                str(self._manifest_path),
                "--job-id",
                job_id,
                "--cancel-file",
                str(cancel_file),
                "--lease",
                str(self._lease_path),
            ]
            try:
                process = self._popen(
                    command,
                    cwd=str(self._repo_root),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                    **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
                )
            except (OSError, ValueError):
                self._snapshot = self._failed_snapshot(
                    code="repair_runner_start_failed",
                    message="环境修复进程无法启动。",
                    retryable=True,
                )
                self._release_lease()
                return self._snapshot.model_copy(deep=True)
            self._process = process
            self._write_lease(process.pid)
            self._heartbeat_stop.clear()
            self._heartbeat = Thread(target=self._heartbeat_loop, name="gs-video-repair-lease", daemon=True)
            self._heartbeat.start()
            self._reader = Thread(target=self._read_runner, args=(job_id, process), name="gs-video-repair-reader", daemon=True)
            self._reader.start()
            return self._snapshot.model_copy(deep=True)

    def cancel(self) -> EnvironmentRepairSnapshot:
        with self._lock:
            if self._snapshot.state not in {
                EnvironmentRepairState.RUNNING,
                EnvironmentRepairState.CANCELLING,
            }:
                return self._snapshot.model_copy(deep=True)
            self._snapshot = self._snapshot.model_copy(
                update={
                    "state": EnvironmentRepairState.CANCELLING,
                    "message": "正在取消环境修复…",
                }
            )
            cancel_file = self._cancel_file
        if cancel_file is not None:
            try:
                cancel_file.write_text("cancel\n", encoding="ascii")
            except OSError:
                pass
        return self.snapshot()

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._shutdown_sync)

    def _shutdown_sync(self) -> None:
        self.cancel()
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=self._CANCEL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        reader = self._reader
        if reader is not None and reader is not current_thread():
            reader.join(timeout=2.0)

    def _claim_lease(self, lease_path: Path, job_id: str) -> None:
        if lease_path.exists():
            try:
                age = time.time() - lease_path.stat().st_mtime
            except OSError:
                age = 0.0
            if age < self._LEASE_STALE_SECONDS:
                raise EnvironmentRepairBusyError("another environment repair is still active")
            lease_path.unlink(missing_ok=True)
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        lease_path.write_text(
            json.dumps({"job_id": job_id, "pid": os.getpid(), "updated_at": time.time()}),
            encoding="utf-8",
        )

    def _write_lease(self, runner_pid: int) -> None:
        self._lease_path.parent.mkdir(parents=True, exist_ok=True)
        job_id = self._snapshot.job_id
        payload = {"job_id": job_id, "pid": runner_pid, "updated_at": time.time()}
        temporary = self._lease_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self._lease_path)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(1.0):
            try:
                with self._lock:
                    process = self._process
                    if process is None or process.poll() is not None:
                        return
                    self._write_lease(process.pid)
            except OSError:
                return

    def _read_runner(self, job_id: str, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is not None:
            for line in stream:
                if len(line) > 128 * 1024:
                    continue
                try:
                    payload = json.loads(line)
                    snapshot = EnvironmentRepairSnapshot.model_validate(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if snapshot.job_id != job_id:
                    continue
                with self._lock:
                    if self._snapshot.job_id == job_id:
                        self._snapshot = snapshot
        try:
            return_code = process.wait()
        except OSError:
            return_code = 1
        self._finish(job_id, return_code)

    def _finish(self, job_id: str, return_code: int) -> None:
        with self._lock:
            if self._snapshot.job_id != job_id:
                return
            current = self._snapshot.model_copy(deep=True)
            cancelled = current.state in {
                EnvironmentRepairState.CANCELLING,
                EnvironmentRepairState.CANCELLED,
            } or return_code == 130
        if cancelled:
            final = current.model_copy(
                update={
                    "state": EnvironmentRepairState.CANCELLED,
                    "progress": min(current.progress, 1.0),
                    "message": "环境修复已取消，可从已下载内容继续。",
                    "resume_available": self._resume_available(),
                }
            )
        elif return_code == 0:
            try:
                report = self._environment_doctor.check()
            except Exception as error:  # pragma: no cover - defensive process boundary
                final = self._failed_snapshot(
                    code="environment_probe_failed",
                    message="环境修复后无法完成环境探针。",
                    retryable=True,
                )
                final = final.model_copy(update={"message": str(error)[:512]})
            else:
                if isinstance(report, EnvironmentReport) and report.ready:
                    final = current.model_copy(
                        update={
                            "state": EnvironmentRepairState.SUCCEEDED,
                            "progress": 1.0,
                            "step": "probe",
                            "message": "环境修复完成，探针已通过。",
                            "environment": report,
                            "resume_available": False,
                        }
                    )
                else:
                    final = self._failed_snapshot(
                        code="environment_not_ready",
                        message="资源已配置，但最终环境探针仍未通过。",
                        retryable=True,
                    ).model_copy(update={"environment": report})
        elif current.state is EnvironmentRepairState.FAILED and current.error is not None:
            final = current.model_copy(
                update={"resume_available": self._resume_available()}
            )
        else:
            final = self._failed_snapshot(
                code="repair_runner_failed",
                message="环境修复进程异常退出。",
                retryable=True,
            )
        with self._lock:
            self._snapshot = final
            self._process = None
            self._heartbeat_stop.set()
            self._release_lease()
            cancel_file = self._cancel_file
            self._cancel_file = None
        if cancel_file is not None:
            cancel_file.unlink(missing_ok=True)
        self._persist_terminal(final)

    def _failed_snapshot(
        self, *, code: str, message: str, retryable: bool
    ) -> EnvironmentRepairSnapshot:
        with self._lock:
            current = self._snapshot
        return current.model_copy(
            update={
                "state": EnvironmentRepairState.FAILED,
                "error": EnvironmentRepairError(code=code, message=message, retryable=retryable),
                "message": message,
                "resume_available": self._resume_available(),
            }
        )

    def _release_lease(self) -> None:
        try:
            self._lease_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _persist_terminal(self, snapshot: EnvironmentRepairSnapshot) -> None:
        path = self._runtime_root / "repair" / "last-result.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                snapshot.model_dump_json(exclude={"environment"}), encoding="utf-8"
            )
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)

    def _resume_available(self) -> bool:
        downloads = self._runtime_root / "downloads"
        try:
            return any(downloads.glob("*.partial"))
        except OSError:
            return False


__all__ = ["EnvironmentRepairBusyError", "EnvironmentRepairManager"]
