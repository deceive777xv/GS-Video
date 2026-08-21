import json
import os
from pathlib import Path

import numpy as np
import pytest

from gs_video.camera.classify import CameraKind
from gs_video.camera.solution import CameraSolution
from gs_video.camera.serialization import (
    MappedTrajectory,
    read_camera_solution,
    read_mapped_trajectory,
    write_camera_solution,
    write_mapped_trajectory,
)


def camera_solution_fixture() -> CameraSolution:
    second_pose = np.eye(4, dtype=np.float64)
    second_pose[:3, 3] = [1.25, -2.5, 3.75]
    return CameraSolution(
        intrinsics=np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        camera_to_world=[np.eye(4, dtype=np.float64), second_pose],
        kind=CameraKind.SIX_DOF,
        confidence=0.875,
        diagnostics={"pairs": [{"tracked": 120, "residual": 0.25}], "note": "ok"},
    )


def test_camera_solution_json_round_trip_preserves_matrices(tmp_path: Path) -> None:
    solution = camera_solution_fixture()
    destination = tmp_path / "camera" / "solution.json"

    write_camera_solution(destination, solution)
    restored = read_camera_solution(destination)

    np.testing.assert_allclose(restored.intrinsics, solution.intrinsics)
    for restored_pose, original_pose in zip(
        restored.camera_to_world, solution.camera_to_world, strict=True
    ):
        np.testing.assert_allclose(restored_pose, original_pose)
    assert restored.kind is solution.kind
    assert restored.confidence == pytest.approx(solution.confidence)
    assert restored.diagnostics == solution.diagnostics
    assert set(json.loads(destination.read_text(encoding="utf-8"))) == {
        "version",
        "intrinsics",
        "frame_intrinsics",
        "camera_to_world",
        "kind",
        "confidence",
        "diagnostics",
        "source_ground",
    }


def test_mapped_trajectory_json_round_trip_preserves_fov_and_poses(
    tmp_path: Path,
) -> None:
    second_pose = np.eye(4, dtype=np.float64)
    second_pose[:3, 3] = [0.5, 0.0, -1.0]
    trajectory = MappedTrajectory(
        fov_y_degrees=58.5,
        camera_to_world=(np.eye(4, dtype=np.float64), second_pose),
    )
    destination = tmp_path / "camera" / "mapped.json"

    write_mapped_trajectory(destination, trajectory)
    restored = read_mapped_trajectory(destination)

    assert restored.fov_y_degrees == pytest.approx(58.5)
    for restored_pose, original_pose in zip(
        restored.camera_to_world, trajectory.camera_to_world, strict=True
    ):
        np.testing.assert_allclose(restored_pose, original_pose)
    assert set(json.loads(destination.read_text(encoding="utf-8"))) == {
        "version",
        "fov_y_degrees",
        "camera_to_world",
    }


def test_mapped_trajectory_round_trip_preserves_per_frame_intrinsics(
    tmp_path: Path,
) -> None:
    trajectory = MappedTrajectory(
        fov_y_degrees=60.0,
        camera_to_world=(np.eye(4), np.eye(4)),
        frame_intrinsics=(
            np.array([[800.0, 0.0, 610.0], [0.0, 805.0, 350.0], [0.0, 0.0, 1.0]]),
            np.array([[820.0, 0.0, 612.0], [0.0, 825.0, 352.0], [0.0, 0.0, 1.0]]),
        ),
        source_size=(1280, 720),
    )
    destination = tmp_path / "camera" / "mapped-v2.json"

    write_mapped_trajectory(destination, trajectory)
    restored = read_mapped_trajectory(destination)

    assert restored.source_size == (1280, 720)
    assert restored.frame_intrinsics is not None
    for actual, expected in zip(
        restored.frame_intrinsics, trajectory.frame_intrinsics, strict=True
    ):
        np.testing.assert_allclose(actual, expected)
    assert set(json.loads(destination.read_text(encoding="utf-8"))) == {
        "version",
        "fov_y_degrees",
        "camera_to_world",
        "frame_intrinsics",
        "source_width",
        "source_height",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("unexpected", True),
        lambda payload: payload.__setitem__("version", 3),
        lambda payload: payload["intrinsics"][0].__setitem__(0, float("nan")),
        lambda payload: payload["intrinsics"][0].__setitem__(0, "900.0"),
        lambda payload: payload["intrinsics"][0].__setitem__(0, True),
        lambda payload: payload["camera_to_world"][0][3].__setitem__(3, 2.0),
        lambda payload: payload.__setitem__("kind", "unknown"),
        lambda payload: payload["diagnostics"].__setitem__("bad", float("inf")),
    ],
)
def test_camera_solution_reader_rejects_noncanonical_or_unsafe_json(
    tmp_path: Path,
    mutation: object,
) -> None:
    destination = tmp_path / "solution.json"
    payload = {
        "version": 1,
        "intrinsics": np.eye(3).tolist(),
        "camera_to_world": [np.eye(4).tolist()],
        "kind": "fixed",
        "confidence": 0.75,
        "diagnostics": {},
    }
    mutation(payload)  # type: ignore[operator]
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((OSError, ValueError)):
        read_camera_solution(destination)


def test_mapped_trajectory_reader_rejects_non_rigid_and_extra_fields(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "mapped.json"
    destination.write_text(
        json.dumps(
            {
                "version": 1,
                "fov_y_degrees": 60.0,
                "camera_to_world": [np.diag([2.0, 1.0, 1.0, 1.0]).tolist()],
                "unexpected": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_mapped_trajectory(destination)


def test_camera_solution_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate.json"
    destination.write_text(
        '{"version":1,"version":1,"intrinsics":[],"camera_to_world":[],"kind":"fixed",'
        '"confidence":0.5,"diagnostics":{}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        read_camera_solution(destination)


def test_camera_reader_rejects_hardlinked_file(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_camera_solution(source, camera_solution_fixture())
    linked = tmp_path / "linked.json"
    os.link(source, linked)

    with pytest.raises(OSError, match="camera JSON"):
        read_camera_solution(linked)


def test_camera_reader_enforces_bounded_input_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gs_video.camera.serialization as serialization

    destination = tmp_path / "oversized.json"
    destination.write_bytes(b"{}" * 33)
    monkeypatch.setattr(serialization, "MAX_CAMERA_JSON_BYTES", 64)

    with pytest.raises(OSError, match="too large"):
        read_camera_solution(destination)


def test_camera_writer_rejects_non_json_diagnostics_without_publishing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "solution.json"
    solution = camera_solution_fixture()
    invalid = CameraSolution(
        solution.intrinsics,
        solution.camera_to_world,
        solution.kind,
        solution.confidence,
        {"unsupported": {1, 2, 3}},
    )

    with pytest.raises((TypeError, ValueError), match="JSON"):
        write_camera_solution(destination, invalid)

    assert not destination.exists()
