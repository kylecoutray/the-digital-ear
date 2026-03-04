import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np

from digital_ear.features import FrameExtractor, hann_window, rfft_mag
from digital_ear.pitch import HPSPitchDetector


def generate_sine(sr: int, freq: float, seconds: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / float(sr)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def run_pitch(sr: int, freq: float, nfft: int = 2048, hop: int = 2048) -> float:
    x = generate_sine(sr, freq)
    fx = FrameExtractor(n_fft=nfft, hop=hop)
    win = hann_window(nfft)
    det = HPSPitchDetector(sr=float(sr), n_fft=nfft, fmin=80.0, fmax=1000.0)

    f0s = []
    for i in range(0, len(x), 2048):
        blk = x[i:i+2048]
        for frame in fx.push(blk):
            mag = rfft_mag(frame, win)
            f0, conf = det.estimate(mag)
            if f0 is not None:
                f0s.append(f0)

    return float(np.median(f0s))


if __name__ == "__main__":
    sr = 44100
    for target in (220.0, 440.0, 880.0):
        med = run_pitch(sr, target)
        print(f"target={target:.1f} median_f0={med:.2f}")

        # Pass condition: within 1 FFT bin
        bin_w = sr / 2048
        assert abs(med - target) <= bin_w, f"f0 {med} too far from {target} (>{bin_w})"

    print("PASS")