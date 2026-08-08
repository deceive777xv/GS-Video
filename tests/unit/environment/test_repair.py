from __future__ import annotations

from pathlib import Path
import subprocess

from gs_video.api.schemas import EnvironmentRepairState
from gs_video.environment.repair import EnvironmentRepairManager


def test_manager_reports_runner_start_failure_without_leaking_the_lease(
    tmp_path: Path,
) -> None:
    def fail_popen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("process creation denied")

    runtime_root = tmp_path / ".runtime"
    runtime_events: list[object] = []
    runtime_token = object()
    manager = EnvironmentRepairManager(
        repo_root=tmp_path,
        runtime_root=runtime_root,
        runtime_config=runtime_root / "desktop-runtime.json",
        environment_doctor=object(),
        popen=fail_popen,  # type: ignore[arg-type]
        acquire_runtime=lambda: runtime_events.append("acquired") or runtime_token,
        release_runtime=lambda token: runtime_events.append(token),
    )

    snapshot = manager.start()

    assert snapshot.state is EnvironmentRepairState.FAILED
    assert snapshot.error is not None
    assert snapshot.error.code == "repair_runner_start_failed"
    assert not (runtime_root / "repair" / "lease.json").exists()
    assert runtime_events == ["acquired", runtime_token]


def test_manager_reaps_spawned_runner_when_lifecycle_initialization_fails(
    tmp_path: Path,
) -> None:
    class SpawnedProcess:
        pid = 4321
        terminated = False
        killed = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if not self.terminated and not self.killed:
                raise subprocess.TimeoutExpired("repair", 2.0)
            return -15

        def kill(self) -> None:
            self.killed = True

    runtime_root = tmp_path / ".runtime"
    process = SpawnedProcess()
    runtime_events: list[str] = []
    manager = EnvironmentRepairManager(
        repo_root=tmp_path,
        runtime_root=runtime_root,
        runtime_config=runtime_root / "desktop-runtime.json",
        environment_doctor=object(),
        popen=lambda *_args, **_kwargs: process,  # type: ignore[arg-type]
        acquire_runtime=lambda: runtime_events.append("acquired") or object(),
        release_runtime=lambda _token: runtime_events.append("released"),
    )

    def fail_write_lease(_pid: int) -> None:
        raise OSError("lease write denied")

    manager._write_lease = fail_write_lease  # type: ignore[method-assign]

    snapshot = manager.start()

    assert snapshot.state is EnvironmentRepairState.FAILED
    assert snapshot.error is not None
    assert snapshot.error.code == "repair_runner_initialize_failed"
    assert process.terminated is True
    assert runtime_events == ["acquired", "released"]
    assert not (runtime_root / "repair" / "lease.json").exists()
