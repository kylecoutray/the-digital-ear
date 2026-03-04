import sys
from pathlib import Path
import numpy as np


from digital_ear.features import FrameExtractor, hann_window, rfft_mag, bin_freq


def generate_sine(sr: int, freq: float, seconds: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / float(sr)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def run_tone_test(sr: int = 44100, freq: float = 440.0, nfft: int = 2048, hop: int = 2048, block: int = 2048):
    x = generate_sine(sr, freq)
    fx = FrameExtractor(n_fft=nfft, hop=hop)
    win = hann_window(nfft)

    peaks = []
    for i in range(0, len(x), block):
        blk = x[i:i+block]
        for frame in fx.push(blk):
            mag = rfft_mag(frame, win)
            k = int(np.argmax(mag))
            peaks.append(bin_freq(k, float(sr), nfft))

    return peaks


if __name__ == "__main__":
    peaks = run_tone_test() # test at 440 Hz
    # Print first few peak estimates
    print("First 5 peak Hz:", [round(p, 2) for p in peaks[:5]])
    # Simple pass condition, median peak is within 1 bin of 440
    # Bin width = sr/nfft
    sr = 44100
    nfft = 2048
    bin_w = sr / nfft
    med = float(np.median(peaks))
    print("Median peak Hz:", round(med, 2), "Bin width:", round(bin_w, 2))
    assert abs(med - 440.0) <= bin_w, f"Median peak {med} too far from 440 (>{bin_w})"
    print("PASS")