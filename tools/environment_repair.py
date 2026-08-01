from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gs_video.environment.download import (  # noqa: E402
    RepairCancelled,
    RepairDownloadError,
    RepairSecurityError,
    download_verified,
    extract_zip,
    sha256_file,
)
from gs_video.environment.manifest import (  # noqa: E402
    DEFAULT_ALLOWED_HOSTS,
    RuntimeAction,
    RuntimeManifest,
    RuntimeManifestError,
    RuntimeResource,
)


class RepairRunnerError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class LeaseExpired(RepairRunnerError):
    def __init__(self) -> None:
        super().__init__("repair_lease_expired", "环境修复管理器已失联，未激活半成品。")


class LeaseGuard:
    def __init__(self, path: Path, job_id: str) -> None:
        self._path = path
        self._job_id = job_id

    def check(self) -> None:
        try:
            if time.time() - self._path.stat().st_mtime > 15.0:
                raise LeaseExpired()
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LeaseExpired() from error
        if payload.get("job_id") != self._job_id:
            raise LeaseExpired()


@dataclass
class Activation:
    target: Path
    backup: Path
    had_previous: bool


@dataclass
class ConfigBackup:
    target: Path
    backup: Path
    had_previous: bool


class RuntimeRepairRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        runtime_root: Path,
        runtime_config: Path,
        manifest_path: Path,
        job_id: str,
        cancel_file: Path,
        lease: Path,
        emit: Callable[..., None],
    ) -> None:
        self.repo_root = repo_root.resolve(strict=False)
        self.runtime_root = runtime_root.resolve(strict=False)
        self.runtime_config = runtime_config.resolve(strict=False)
        self.manifest_path = manifest_path.resolve(strict=False)
        self.job_id = job_id
        self.cancel_file = cancel_file
        self.lease = LeaseGuard(lease, job_id)
        self.emit = emit
        self.downloaded_bytes = 0
        self.total_bytes: int | None = None
        self.staging_root = self.runtime_root / "staging" / job_id
        self.download_root = self.runtime_root / "downloads"
        self.backup_root = self.runtime_root / "repair" / "backups" / job_id

    def run(self) -> int:
        activations: list[Activation] = []
        config_backup: ConfigBackup | None = None
        try:
            self._preflight()
            manifest = RuntimeManifest.load(self.manifest_path)
            sizes = [resource.size for resource in manifest.resources]
            known_sizes = [size for size in sizes if size is not None]
            self.total_bytes = sum(known_sizes) if len(known_sizes) == len(sizes) else None
            staged: dict[Path, Path] = {}
            self._download_resources(manifest, staged)
            editable_actions = self._run_actions(manifest, staged)
            config_backup = self._backup_runtime_config()
            activations = self._activate(staged)
            self._run_editable_actions(editable_actions)
            self._refresh_runtime_config()
            self._check()
            self.emit(
                state="succeeded",
                step="probe",
                progress=1.0,
                message="资源已安装，等待应用完成环境探针。",
                downloaded_bytes=self.downloaded_bytes,
                total_bytes=self.total_bytes,
            )
            return 0
        except RepairCancelled:
            if activations:
                self._rollback(activations)
            if config_backup is not None:
                self._restore_runtime_config(config_backup)
            self.emit(
                state="cancelled",
                step="cancelled",
                progress=0.0,
                message="环境修复已取消，可从已下载内容继续。",
                downloaded_bytes=self.downloaded_bytes,
                total_bytes=self.total_bytes,
            )
            return 130
        except (RepairRunnerError, RepairDownloadError, RuntimeManifestError, OSError, ValueError) as error:
            if activations:
                self._rollback(activations)
            if config_backup is not None:
                self._restore_runtime_config(config_backup)
            code = getattr(error, "code", "repair_failed")
            message = getattr(error, "message", str(error))[:512]
            self.emit(
                state="failed",
                step="failed",
                progress=0.0,
                message=message,
                error={"code": code, "message": message, "retryable": True},
                downloaded_bytes=self.downloaded_bytes,
                total_bytes=self.total_bytes,
            )
            return 1
        finally:
            self._cleanup_staging()

    def _preflight(self) -> None:
        self._check()
        manifest = RuntimeManifest.load(self.manifest_path)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.download_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        sizes = [resource.size for resource in manifest.resources]
        required = sum(size for size in sizes if size is not None)
        free = shutil.disk_usage(self.runtime_root).free
        if required > free:
            raise RepairRunnerError(
                "disk_space_insufficient",
                "工程所在磁盘空间不足，无法准备环境修复。",
            )
        self.emit(
            state="running",
            step="preflight",
            progress=0.0,
            message="清单和磁盘空间检查通过。",
            downloaded_bytes=0,
            total_bytes=required,
        )

    def _download_resources(
        self, manifest: RuntimeManifest, staged: dict[Path, Path]
    ) -> None:
        downloaded_before = 0
        for resource in manifest.resources:
            self._check()
            if self._resource_already_active(resource):
                downloaded_before += resource.size or 0
                self.downloaded_bytes = downloaded_before
                self.emit(
                    state="running",
                    step="download",
                    resource_id=resource.id,
                    resource_name=resource.version,
                    progress=(
                        downloaded_before / self.total_bytes if self.total_bytes else 1.0
                    ),
                    message=f"已复用 {resource.id}。",
                    downloaded_bytes=downloaded_before,
                    total_bytes=self.total_bytes,
                )
                continue
            download_path = self.download_root / resource.id
            self.emit(
                state="running",
                step="download",
                resource_id=resource.id,
                resource_name=resource.version,
                progress=(downloaded_before / self.total_bytes if self.total_bytes else 0.0),
                message=f"正在下载 {resource.id}…",
                downloaded_bytes=downloaded_before,
                total_bytes=self.total_bytes,
            )

            def on_progress(current: int, _total: int | None) -> None:
                self._check()
                self.downloaded_bytes = downloaded_before + current
                self.emit(
                    state="running",
                    step="download",
                    resource_id=resource.id,
                    resource_name=resource.version,
                    progress=(
                        self.downloaded_bytes / self.total_bytes if self.total_bytes else 0.0
                    ),
                    message=f"正在下载 {resource.id}…",
                    downloaded_bytes=self.downloaded_bytes,
                    total_bytes=self.total_bytes,
                )

            for attempt in range(3):
                try:
                    download_verified(
                        resource.url,
                        download_path,
                        expected_size=resource.size,
                        expected_sha256=resource.sha256,
                        progress=on_progress,
                        cancelled=self._cancelled,
                        allowed_hosts=DEFAULT_ALLOWED_HOSTS,
                    )
                    break
                except RepairCancelled:
                    raise
                except RepairDownloadError:
                    if attempt == 2:
                        raise
                    time.sleep(0.25 * (attempt + 1))
            downloaded_before += resource.size or download_path.stat().st_size
            self.downloaded_bytes = downloaded_before
            staged_target = self._stage_resource(resource, download_path)
            staged[self._resolve(resource.target)] = staged_target

    def _resource_already_active(self, resource: RuntimeResource) -> bool:
        target = self._resolve(resource.target)
        if resource.kind == "file":
            if not target.is_file():
                return False
            if resource.size is not None and target.stat().st_size != resource.size:
                return False
            return sha256_file(target) == resource.sha256
        return target.is_dir() and all(
            (target / Path(*marker.parts)).is_file() for marker in resource.markers
        )

    def _stage_resource(self, resource: RuntimeResource, download_path: Path) -> Path:
        if resource.kind == "file":
            staged = self.staging_root / Path(*resource.target.parts)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(download_path, staged)
            if (
                (resource.size is not None and staged.stat().st_size != resource.size)
                or sha256_file(staged) != resource.sha256
            ):
                raise RepairRunnerError("resource_stage_integrity_failed", "资源暂存校验失败。")
            return staged
        assert resource.extract_to is not None
        staged = self.staging_root / Path(*resource.extract_to.parts)
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True, exist_ok=True)
        extract_zip(download_path, staged, strip_prefix=resource.strip_prefix)
        return staged

    def _run_actions(
        self, manifest: RuntimeManifest, staged: dict[Path, Path]
    ) -> tuple[RuntimeAction, ...]:
        editable: list[RuntimeAction] = []
        for action in manifest.actions:
            self._check()
            if action.type == "create_venv":
                assert action.target is not None
                existing_target = self._resolve(action.target)
                existing_python = existing_target / "Scripts" / "python.exe"
                if existing_python.is_file():
                    continue
                target = self._stage_directory(action.target, staged)
                base_python = self.repo_root / ".venv" / "Scripts" / "python.exe"
                if not base_python.is_file():
                    raise RepairRunnerError(
                        "project_python_missing",
                        "主 Python 环境不存在，无法创建 worker 环境。",
                        retryable=False,
                    )
                self._run([str(base_python), "-m", "venv", str(target)])
            elif action.type == "install_packages":
                assert action.target is not None
                target = self._stage_directory(action.target, staged)
                python = self._venv_python(target)
                if self._packages_satisfied(python, action.packages):
                    continue
                command = ["uv", "pip", "install", "--python", str(python)]
                if action.index_url is not None:
                    command.extend(["--index-url", action.index_url])
                command.extend(action.packages)
                self._run(command)
            elif action.type == "install_editable":
                editable.append(action)
        return tuple(editable)

    def _run_editable_actions(self, actions: tuple[RuntimeAction, ...]) -> None:
        for action in actions:
            self._check()
            assert action.target is not None
            python = self._venv_python(self._resolve(action.target))
            source = self.repo_root if action.source is None else self._resolve(action.source)
            if not source.is_dir():
                raise RepairRunnerError(
                    "editable_source_missing",
                    f"editable 安装源不存在: {source}",
                )
            self._run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--no-deps",
                    "--editable",
                    str(source),
                ]
            )

    def _stage_directory(self, relative: Any, staged: dict[Path, Path]) -> Path:
        target = self._resolve(relative)
        existing = staged.get(target)
        if existing is not None:
            return existing
        if target.is_dir():
            return target
        stage = self.staging_root / Path(*relative.parts)
        if stage.exists():
            shutil.rmtree(stage)
        stage.parent.mkdir(parents=True, exist_ok=True)
        staged[target] = stage
        return stage

    def _venv_python(self, root: Path) -> Path:
        python = root / "Scripts" / "python.exe"
        if not python.is_file():
            raise RepairRunnerError("worker_python_missing", f"worker Python 未生成: {python}")
        return python

    def _packages_satisfied(self, python: Path, packages: tuple[str, ...]) -> bool:
        names: list[str] = []
        for package in packages:
            name = package.split("==", 1)[0].strip()
            if not name:
                return False
            names.append(name)
        script = (
            "import importlib.metadata as m; "
            "names = "
            + repr(names)
            + "; "
            "print('\\n'.join(m.version(name) for name in names))"
        )
        try:
            options: dict[str, Any] = {
                "cwd": str(self.repo_root),
                "capture_output": True,
                "text": True,
                "timeout": 30,
                "shell": False,
                "check": False,
            }
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            result = cast(
                subprocess.CompletedProcess[str],
                subprocess.run([str(python), "-c", script], **options),
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode != 0:
            return False
        installed = result.stdout.splitlines()
        expected = [package.split("==", 1)[1].strip() for package in packages]
        return installed == expected

    def _activate(self, staged: dict[Path, Path]) -> list[Activation]:
        activations: list[Activation] = []
        try:
            for target, stage in sorted(staged.items(), key=lambda item: len(item[0].parts)):
                self._check()
                if not stage.exists():
                    raise RepairRunnerError("staged_resource_missing", f"暂存资源不存在: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = self.backup_root / Path(*target.relative_to(self.runtime_root).parts)
                backup.parent.mkdir(parents=True, exist_ok=True)
                had_previous = target.exists()
                if had_previous:
                    os.replace(target, backup)
                os.replace(stage, target)
                activations.append(Activation(target, backup, had_previous))
            return activations
        except Exception:
            self._rollback(activations)
            raise

    def _rollback(self, activations: list[Activation]) -> None:
        for activation in reversed(activations):
            try:
                if activation.target.is_dir() and not activation.target.is_symlink():
                    shutil.rmtree(activation.target)
                else:
                    activation.target.unlink(missing_ok=True)
                if activation.had_previous:
                    os.replace(activation.backup, activation.target)
            except OSError:
                pass

    def _refresh_runtime_config(self) -> None:
        prepare = self.repo_root / "tools" / "prepare_desktop_runtime.py"
        if not prepare.is_file():
            return
        self._run(
            [
                sys.executable,
                str(prepare),
                "--repo-root",
                str(self.repo_root),
                "--allow-missing-resources",
            ]
        )

    def _backup_runtime_config(self) -> ConfigBackup:
        backup = self.backup_root / "desktop-runtime.json"
        target = self.runtime_config
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise RepairSecurityError("runtime configuration is not an ordinary file")
            shutil.copyfile(target, backup)
            return ConfigBackup(target=target, backup=backup, had_previous=True)
        return ConfigBackup(target=target, backup=backup, had_previous=False)

    def _restore_runtime_config(self, backup: ConfigBackup) -> None:
        try:
            if backup.had_previous:
                os.replace(backup.backup, backup.target)
            else:
                backup.target.unlink(missing_ok=True)
        except OSError:
            pass

    def _run(self, command: list[str]) -> None:
        self._check()
        try:
            options: dict[str, Any] = {
                "cwd": str(self.repo_root),
                "capture_output": True,
                "text": True,
                "timeout": 30 * 60,
                "shell": False,
                "check": False,
            }
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = cast(
                subprocess.CompletedProcess[str],
                subprocess.run(command, **options),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RepairRunnerError("install_command_failed", "环境安装命令无法执行。") from error
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "环境安装命令失败").strip()
            raise RepairRunnerError("install_command_failed", message[:512])

    def _refresh_lease(self) -> None:
        self.lease.check()

    def _check(self) -> None:
        if self._cancelled():
            raise RepairCancelled("runtime repair was cancelled")
        self._refresh_lease()

    def _cancelled(self) -> bool:
        return self.cancel_file.exists()

    def _resolve(self, relative: Any) -> Path:
        if not hasattr(relative, "parts"):
            raise RepairRunnerError("invalid_manifest_path", "清单路径无效。", retryable=False)
        candidate = self.runtime_root / Path(*relative.parts)
        resolved = candidate.resolve(strict=False)
        if resolved != self.runtime_root and self.runtime_root not in resolved.parents:
            raise RepairSecurityError("清单目标路径逃逸出工程 runtime")
        return resolved

    def _cleanup_staging(self) -> None:
        try:
            if self.staging_root.exists():
                shutil.rmtree(self.staging_root)
        except OSError:
            pass


def _emit_factory(job_id: str) -> Callable[..., None]:
    def emit(**payload: Any) -> None:
        payload = {"job_id": job_id, **payload}
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)

    return emit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair the GS Video project runtime")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cancel-file", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = RuntimeRepairRunner(
        repo_root=args.repo_root,
        runtime_root=args.runtime_root,
        runtime_config=args.runtime_config,
        manifest_path=args.manifest,
        job_id=args.job_id,
        cancel_file=args.cancel_file,
        lease=args.lease,
        emit=_emit_factory(args.job_id),
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
