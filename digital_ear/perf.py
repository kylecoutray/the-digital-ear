from __future__ import annotations
import os
import time
import psutil #read process memory

# timing / memory helper 

class PerfLogger:
    """Tracks elapsed time and peak RSS in MB.
    Used to make sure we stay within Pi memory limits."""

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self._t0 = 0.0
        self._peak_rss = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self.sample()

    def sample(self) -> None:
        rss = self._proc.memory_info().rss  # bytes
        if rss > self._peak_rss:
            self._peak_rss = rss #storing peak rss

    def stop_and_report(self) -> tuple[float, float]:
        self.sample()
        elapsed_s = time.perf_counter() - self._t0
        peak_mb = self._peak_rss / (1024 * 1024)
        return elapsed_s, peak_mb