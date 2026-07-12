from threading import Event

from gs_video.domain.errors import CancelledError


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("任务已取消")
