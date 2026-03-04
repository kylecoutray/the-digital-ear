import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from digital_ear.features import FrameExtractor, hann_window, rfft_mag
from digital_ear.pitch import HPSPitchDetector


def run_on_signal(x: np.ndarray, sr: int = 44100) -> list[tuple[float | None, float]]:
    nfft = 2048
    fx = FrameExtractor(n_fft=nfft, hop=nfft)
    win = hann_window(nfft)
    det = HPSPitchDetector(sr=float(sr), n_fft=nfft, fmin=80.0, fmax=1000.0)

    out = []
    for i in range(0, len(x), 2048):
        blk = x[i:i+2048].astype(np.float32, copy=False)
        for frame in fx.push(blk):
            mag = rfft_mag(frame, win)
            f0, conf = det.estimate(mag)
            out.append((f0, conf))
    return out


if __name__ == "__main__":
    sr = 44100
    dur = 2.0
    n = int(sr * dur)

    # Silence
    sil = np.zeros(n, dtype=np.float32)
    r1 = run_on_signal(sil, sr)
    voiced1 = sum(1 for f0, _ in r1 if f0 is not None)
    print("silence voiced frames:", voiced1, "of", len(r1))
    assert voiced1 == 0

    # White noise (low amplitude)
    rng = np.random.default_rng(0)
    noise = (0.02 * rng.standard_normal(n)).astype(np.float32)
    r2 = run_on_signal(noise, sr)
    voiced2 = sum(1 for f0, _ in r2 if f0 is not None)
    print("noise voiced frames:", voiced2, "of", len(r2))
    assert voiced2 <= max(1, int(0.05 * len(r2)))  # allow a few false positives

    print("PASS")