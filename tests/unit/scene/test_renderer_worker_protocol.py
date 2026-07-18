from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import gs_video.scene.worker as worker_module
from gs_video.scene.worker_protocol import (
    MAX_EVENT_BYTES,
    MAX_REQUEST_BYTES,
    OrbitCameraPayload,
    ProbeRequest,
    RenderPickRequest,
    RenderSequenceRequest,
    parse_worker_event,
    read_worker_request,
    write_worker_request,
    ProbeEvent,
)


def _sequence_request(tmp_path: Path) -> RenderSequenceRequest:
    return RenderSequenceRequest(
        type="render_sequence",
        scene_path=(tmp_path / "scene.ply").absolute(),
        camera_manifest=(tmp_path / "trajectory.json").absolute(),
        output_dir=(tmp_path / "frames").absolute(),
        width=64,
        height=36,
        sh_degree=1,
        background=(0.0, 0.0, 0.0),
        preview_stride=1,
    )


def test_request_round_trip_is_strict_and_bounded(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request = _sequence_request(tmp_path)

    write_worker_request(request_path, request)

    assert read_worker_request(request_path) == request
    assert request_path.stat().st_size < MAX_REQUEST_BYTES


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "probe", "unknown": True},
        {
            "type": "render_sequence",
            "scene_path": "relative.ply",
            "camera_manifest": "relative.json",
            "output_dir": "frames",
            "width": 64,
            "height": 36,
            "sh_degree": 1,
            "background": [0.0, float("nan"), 0.0],
            "preview_stride": 1,
        },
    ],
)
def test_request_rejects_unknown_keys_relative_paths_and_nonfinite_values(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises((ValueError, ValidationError)):
        read_worker_request(request_path)


def test_request_rejects_duplicate_keys_and_oversized_file(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"type":"probe","type":"probe"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_worker_request(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_REQUEST_BYTES + 1))
    with pytest.raises(ValueError, match="16 MiB"):
        read_worker_request(oversized)


def test_pick_payload_rejects_nonfinite_camera_and_boolean_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        OrbitCameraPayload(
            target=(0.0, 0.0, float("inf")),
            distance=1.0,
            yaw=0.0,
            pitch=0.0,
            fov_y_degrees=60.0,
        )
    with pytest.raises(ValidationError):
        RenderPickRequest(
            type="render_pick",
            scene_path=(tmp_path / "scene.ply").absolute(),
            output_npz=(tmp_path / "pick.npz").absolute(),
            camera=OrbitCameraPayload(
                target=(0.0, 0.0, 1.0),
                distance=1.0,
                yaw=0.0,
                pitch=0.0,
                fov_y_degrees=60.0,
            ),
            width=True,
            height=36,
        )


def test_event_parser_rejects_unknown_nonfinite_duplicate_and_oversized_events() -> None:
    assert parse_worker_event(
        b'{"type":"progress","current":1,"total":2,"message":"rendering"}\n'
    ).type == "progress"
    invalid = (
        b'{"type":"progress","current":1,"total":2,"message":"x","extra":1}\n',
        b'{"type":"progress","current":1,"total":2,"message":NaN}\n',
        b'{"type":"probe","torch":"2","torch":"3","gsplat":"1","device":"cuda"}\n',
        b"x" * (MAX_EVENT_BYTES + 1),
    )
    for payload in invalid:
        with pytest.raises((ValueError, ValidationError)):
            parse_worker_event(payload)


def test_probe_request_has_no_token_field() -> None:
    request = ProbeRequest(type="probe")
    assert request.model_dump() == {"type": "probe"}


def test_worker_redirects_dependency_stdout_away_from_jsonl_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path = tmp_path / "request.json"
    gate = tmp_path / "gate"
    write_worker_request(request_path, ProbeRequest(type="probe"))
    gate.write_text("RELEASE\n", encoding="utf-8")

    def noisy_dependency(_request: object) -> ProbeEvent:
        print("dependency noise")
        return ProbeEvent(type="probe", torch="2.7", gsplat="1.5", device="cuda")

    monkeypatch.setattr(worker_module, "_run_validated_request", noisy_dependency)
    assert worker_module.main([
        "--request", str(request_path), "--startup-gate", str(gate)
    ]) == 0

    captured = capsys.readouterr()
    assert "dependency noise" not in captured.out
    assert captured.err.strip() == "dependency noise"
    assert json.loads(captured.out) == {
        "type": "probe", "torch": "2.7", "gsplat": "1.5", "device": "cuda"
    }
