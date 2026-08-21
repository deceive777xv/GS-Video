from pathlib import Path

from gs_video.camera.vipe_solver import VipeCameraSolver
from gs_video.segmentation.paths import worker_path


def test_vipe_solver_translates_camera_paths_for_wsl(tmp_path: Path) -> None:
    prefix = (
        "wsl.exe",
        "-d",
        "Ubuntu",
        "--",
        "/mnt/e/Project/GS-Video/.runtime/camera/wsl/.venv/bin/python",
    )
    solver = VipeCameraSolver(prefix)
    frame_dir = tmp_path / "frames"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "output"

    command = solver._command(frame_dir, mask_dir, output_dir, 430)

    assert command[:7] == [
        *prefix,
        "-m",
        "gs_video.camera.vipe_worker",
    ]
    assert command[command.index("--frames") + 1] == worker_path(frame_dir, prefix)
    assert command[command.index("--masks") + 1] == worker_path(mask_dir, prefix)
    assert command[command.index("--output") + 1] == worker_path(output_dir, prefix)
    assert command[command.index("--count") + 1] == "430"
    assert "\\" not in " ".join(command)


def test_vipe_solver_exports_project_cache_paths_through_wslenv(
    tmp_path: Path, monkeypatch
) -> None:
    prefix = ("wsl.exe", "--", "/opt/vipe/bin/python")
    cache_root = tmp_path / "camera-cache"
    monkeypatch.setenv("WSLENV", "EXISTING")
    solver = VipeCameraSolver(prefix, cache_root=cache_root)

    environment = solver._environment()

    assert environment["HF_HOME"] == str(cache_root / "huggingface")
    assert environment["HF_HUB_CACHE"] == str(cache_root / "huggingface" / "hub")
    assert environment["TORCH_HOME"] == str(cache_root / "torch")
    assert environment["XDG_CACHE_HOME"] == str(cache_root)
    assert environment["WSLENV"] == (
        "EXISTING:HF_HOME/p:HF_HUB_CACHE/p:TORCH_HOME/p:XDG_CACHE_HOME/p"
    )
