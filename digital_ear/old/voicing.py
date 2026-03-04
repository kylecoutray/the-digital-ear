from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VoicingGate:
    conf_threshold: float = 3.0
    rms_threshold: float = 1e-3  # tune if needed

    def apply(self, f0_hz: float | None, conf: float, frame_rms: float) -> tuple[float | None, bool]:
        """
        Returns (f0_or_none, voiced_bool).
        """
        if f0_hz is None:
            return None, False
        if conf < self.conf_threshold:
            return None, False
        if frame_rms < self.rms_threshold:
            return None, False
        return f0_hz, True