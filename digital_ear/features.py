# iteration 3: frame analysis, windowed FFT

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def hann_window(n: int) -> np.ndarray:
    # float32 Hann window
    return np.hanning(n).astype(np.float32)


def rfft_mag(frame: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    frame: shape (N,), float32
    window: shape (N,), float32
    returns: magnitude spectrum, shape (N//2 + 1,), float32
    """
    xw = frame * window
    X = np.fft.rfft(xw)
    mag = np.abs(X).astype(np.float32)
    return mag


def bin_freq(k: int, sr: float, n_fft: int) -> float:
    return (k * sr) / n_fft


@dataclass
class FrameExtractor:
    """
    Streaming frame extractor with overlap support.
    You push in blocks (any length <= block_size), and it yields frames of length n_fft
    spaced by hop samples.

    Works even if hop < block size (it will yield multiple frames per block as data accumulates).
    """
    n_fft: int
    hop: int
    _buf: np.ndarray = None
    _fill: int = 0

    def __post_init__(self) -> None:
        if self.hop <= 0:
            raise ValueError("hop must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop > self.n_fft:
            raise ValueError("hop must be <= n_fft")
        self._buf = np.zeros(self.n_fft, dtype=np.float32)
        self._fill = 0

    def push(self, x: np.ndarray):
        """
        x: float32 samples
        yields: frames (np.ndarray float32 length n_fft)
        """
        if x.size == 0:
            return
        x = x.astype(np.float32, copy=False)

        idx = 0
        n = x.shape[0]

        while idx < n:
            # how much we can copy into buffer
            space = self.n_fft - self._fill
            take = min(space, n - idx)

            self._buf[self._fill:self._fill + take] = x[idx:idx + take]
            self._fill += take
            idx += take

            # if we have a full frame, yield it, then shift by hop
            while self._fill == self.n_fft:
                frame = self._buf.copy()  # small fixed copy, predictable
                yield frame

                # shift left by hop
                if self.hop == self.n_fft:
                    # no overlap: reset buffer
                    self._fill = 0
                else:
                    remain = self.n_fft - self.hop
                    self._buf[:remain] = self._buf[self.hop:]
                    self._fill = remain