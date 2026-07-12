from gs_video.environment.doctor import EnvironmentDoctor


def test_doctor_reports_missing_commands_without_starting_gpu() -> None:
    doctor = EnvironmentDoctor(which=lambda name: None, cuda_probe=lambda: (False, 0))
    report = doctor.check()
    assert report.ready is False
    assert {issue.code for issue in report.issues} == {
        "ffmpeg_missing",
        "ffprobe_missing",
        "cuda_unavailable",
    }
