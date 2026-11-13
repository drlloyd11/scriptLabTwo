"""Optional, per-search elapsed-time measurements."""

from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Iterator, Optional


class SearchTimings:
    def __init__(self) -> None:
        self.seconds: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount


@contextmanager
def measure(timings: Optional[SearchTimings], name: str) -> Iterator[None]:
    if timings is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        timings.seconds[name] = timings.seconds.get(name, 0.0) + perf_counter() - started
