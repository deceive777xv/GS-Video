from __future__ import annotations

from pathlib import Path

from gs_video.api.schemas import EnvironmentRepairState
from gs_video.environment.repair import EnvironmentRepairManager


def test_manager_reports_runner_start_failure_without_leaking_the_lease(
    tmp_path: Path,
) -> None:
    def fail_popen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("process creation denied")

    runtime_root = tmp_path / ".runtime"
    manager = EnvironmentRepairManager(
        repo_root=tmp_path,
        runtime_root=runtime_root,
        runtime_config=runtime_root / "desktop-runtime.json",
        environment_doctor=object(),
        popen=fail_popen,  # type: ignore[arg-type]
    )

    snapshot = manager.start()

    assert snapshot.state is EnvironmentRepairState.FAILED
    assert snapshot.error is not None
    assert snapshot.error.code == "repair_runner_start_failed"
    assert not (runtime_root / "repair" / "lease.json").exists()
