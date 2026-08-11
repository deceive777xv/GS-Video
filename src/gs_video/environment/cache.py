from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from gs_video.environment.doctor import EnvironmentReport


class EnvironmentReportCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self._report: EnvironmentReport | None = None

    def get(self, probe: Callable[[], EnvironmentReport]) -> EnvironmentReport:
        with self._lock:
            if self._report is None:
                self._report = probe()
            return self._report

    def refresh(self, probe: Callable[[], EnvironmentReport]) -> EnvironmentReport:
        with self._lock:
            report = probe()
            self._report = report
            return report


__all__ = ["EnvironmentReportCache"]
