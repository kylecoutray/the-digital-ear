from __future__ import annotations

from dataclasses import dataclass
import numpy as np

#iteration 2: 
# LPF, HPF, and DC blocker implemented as streaming one-pole filters.

def rms(x: np.ndarray) -> float:
    # stable RMS for float32 vectors
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


@dataclass
class OnePoleLPF:
    """One-pole IIR low-pass, streaming.
    y[n] = y[n-1] + a*(x[n] - y[n-1]), a = 1 - exp(-2*pi*fc/fs)"""
    fs: float
    fc: float
    y1: float = 0.0  # previous output

    def set_params(self, fs: float, fc: float) -> None:
        self.fs = fs
        self.fc = fc

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32, copy=False)

        a = 1.0 - np.exp(-2.0 * np.pi * (self.fc / self.fs))
        y = np.empty_like(x, dtype=np.float32)

        y_prev = float(self.y1)
        for i in range(x.shape[0]):
            y_prev = y_prev + a * (float(x[i]) - y_prev)
            y[i] = y_prev

        self.y1 = y_prev
        return y


@dataclass
class OnePoleHPF:
    """High-pass from one-pole LP: hp(x) = x - lp(x).
    Acts as DC blocker / rumble removal."""
    fs: float
    fc: float
    _lp: OnePoleLPF = None  # created in __post_init__

    def __post_init__(self) -> None:
        self._lp = OnePoleLPF(self.fs, self.fc)

    def set_params(self, fs: float, fc: float) -> None:
        self.fs = fs
        self.fc = fc
        self._lp.set_params(fs, fc)

    def process(self, x: np.ndarray) -> np.ndarray:
        lp = self._lp.process(x)
        # high-pass = original - low-pass
        return (x.astype(np.float32, copy=False) - lp).astype(np.float32, copy=False)


@dataclass
class Preprocessor:
    """Preprocessing chain: DC blocker -> HPF (~70 Hz) -> LPF (~1200 Hz).
    All streaming, stateful across blocks."""
    fs: float
    dc_fc: float = 30.0
    hp_fc: float = 70.0
    lp_fc: float = 1200.0

    _dc: OnePoleHPF = None
    _hp: OnePoleHPF = None
    _lp: OnePoleLPF = None

    def __post_init__(self) -> None:
        self._dc = OnePoleHPF(self.fs, self.dc_fc)
        self._hp = OnePoleHPF(self.fs, self.hp_fc)
        self._lp = OnePoleLPF(self.fs, self.lp_fc)

    def process(self, x: np.ndarray) -> np.ndarray:
        # x assumed float32 [-1,1]
        y = self._dc.process(x)
        y = self._hp.process(y)
        y = self._lp.process(y)
        return y