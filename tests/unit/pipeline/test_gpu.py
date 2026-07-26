from __future__ import annotations

from threading import Event, Thread

from gs_video.domain.errors import CancelledError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.gpu import GpuAdmissionGate


def test_gpu_gate_serializes_callers() -> None:
    gate = GpuAdmissionGate()
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def first() -> None:
        with gate.hold(CancellationToken()):
            first_entered.set()
            assert release_first.wait(1)

    def second() -> None:
        assert first_entered.wait(1)
        with gate.hold(CancellationToken()):
            second_entered.set()

    first_thread = Thread(target=first)
    second_thread = Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(1)
    assert not second_entered.wait(0.1)
    release_first.set()
    first_thread.join(1)
    second_thread.join(1)
    assert second_entered.is_set()


def test_gpu_gate_wait_is_cancellable() -> None:
    gate = GpuAdmissionGate()
    blocker = CancellationToken()
    waiting = CancellationToken()
    entered = Event()
    released = Event()
    failure: list[BaseException] = []

    def first() -> None:
        with gate.hold(blocker):
            entered.set()
            assert released.wait(1)

    def second() -> None:
        try:
            with gate.hold(waiting):
                raise AssertionError("cancelled caller entered the GPU gate")
        except BaseException as error:
            failure.append(error)

    first_thread = Thread(target=first)
    second_thread = Thread(target=second)
    first_thread.start()
    assert entered.wait(1)
    second_thread.start()
    waiting.cancel()
    second_thread.join(1)
    released.set()
    first_thread.join(1)
    assert len(failure) == 1
    assert isinstance(failure[0], CancelledError)
