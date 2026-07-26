from __future__ import annotations

from contextlib import contextmanager
from threading import Event, Lock
from typing import Iterator

from gs_video.domain.errors import CancelledError
from gs_video.pipeline.cancellation import CancellationToken


class GpuAdmissionGate:
    """Serialize GPU worker lifetimes and keep queued callers cancellable."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._closed = Event()

    @contextmanager
    def hold(self, token: CancellationToken) -> Iterator[None]:
        acquired = False
        while not acquired:
            token.raise_if_cancelled()
            if self._closed.is_set():
                raise CancelledError("GPU worker registry is shutting down")
            acquired = self._lock.acquire(timeout=0.05)
        try:
            token.raise_if_cancelled()
            if self._closed.is_set():
                raise CancelledError("GPU worker registry is shutting down")
            yield
        finally:
            self._lock.release()

    def close(self) -> None:
        self._closed.set()


__all__ = ["GpuAdmissionGate"]
