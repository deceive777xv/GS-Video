from collections.abc import Callable


ProgressEmitter = Callable[[int, int, str], None]


def discard_progress(current: int, total: int, message: str) -> None:
    del current, total, message
