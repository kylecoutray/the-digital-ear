# pitch.py — FFT + HPS pitch detector, Arduino-inspired, production-tuned
#
# WHAT WAS WRONG WITH THE PREVIOUS VERSION:
#   1. Harmonic scoring used weighted linear sum:
#        score(k) = mag[k] + mag[2k]/2 + mag[3k]/3
#      When a harmonic is LOUDER than the fundamental (common in real instruments),
#      that harmonic bin outscores the fundamental and we pick the wrong octave.
#
#   2. Fix: use HPS in LOG domain (sum of logs = log of products).
#      HPS multiplies the spectrum with downsampled copies of itself.
#      A bin only scores high if ALL its harmonic positions also have energy.
#      This strongly locks onto the fundamental even when harmonics are louder.
#
#   3. Confidence = HPS_peak / HPS_median gives a much better voiced/unvoiced
#      signal than raw mag peak/median.

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def _build_midi_freq_table() -> np.ndarray:
    notes = np.arange(128, dtype=np.float64)
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))

_MIDI_FREQS = _build_midi_freq_table()


def hz_to_nearest_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(np.argmin(np.abs(_MIDI_FREQS - hz)))


@dataclass
class FFTPitchDetector:
    """
    FFT + HPS (Harmonic Product Spectrum) monophonic pitch detector.

    Pipeline:
      1. Hamming window the frame
      2. rFFT → log magnitude spectrum
      3. HPS: sum log-mag with log-mag downsampled by 2, 3, 4 ...
         (equivalent to multiplying the spectra — strong peaks at the fundamental)
      4. Find peak of HPS in [fmin, fmax] band
      5. Parabolic interpolation for sub-bin accuracy
      6. Confidence = HPS_peak / HPS_median

    threshold param is unused (kept for API compat with old YIN imports).
    Use conf_th in main.py to gate voiced/unvoiced.
    """
    sr: float
    n_fft: int
    fmin: float = 80.0
    fmax: float = 2000.0
    n_harmonics: int = 4      # number of harmonics to multiply (3-5 is good)
    threshold: float = 0.0    # unused, kept for API compat

    _win: np.ndarray   = field(init=False, repr=False)
    _k_min: int        = field(init=False, repr=False)
    _k_max: int        = field(init=False, repr=False)
    # k_max clamped so k*n_harmonics never exceeds n_bins
    _k_max_hps: int    = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._win = np.hamming(self.n_fft).astype(np.float32)
        n_bins = self.n_fft // 2 + 1
        self._k_min = max(1, int(np.floor(self.fmin * self.n_fft / self.sr)))
        self._k_max = min(n_bins - 1, int(np.ceil(self.fmax * self.n_fft / self.sr)))
        # HPS requires mag[k * (n_harmonics+1)] to exist
        self._k_max_hps = min(self._k_max, (n_bins - 1) // (self.n_harmonics + 1))
        if self._k_max_hps < self._k_min:
            raise ValueError(
                f"FFTPitchDetector: fmax too high or n_harmonics too large "
                f"for n_fft={self.n_fft}, sr={self.sr}"
            )

    def estimate(self, frame: np.ndarray) -> tuple[float | None, float]:
        """
        frame : shape (n_fft,), float32 time-domain samples
        returns: (f0_hz or None, confidence)
          confidence = HPS_peak / HPS_median in the search band.
          For voiced frames this is typically > 8.
          For noise/silence it's typically < 4.
        """
        if frame.size < self.n_fft:
            return None, 0.0

        x = frame[:self.n_fft].astype(np.float32)

        # --- Hamming window ---
        xw = x * self._win

        # --- Log magnitude spectrum ---
        mag = np.abs(np.fft.rfft(xw)).astype(np.float64)
        mag = np.maximum(mag, 1e-12)          # avoid log(0)
        log_mag = np.log(mag)                  # log domain → sums become products

        # --- HPS in log domain ---
        # hps[k] = log_mag[k] + log_mag[2k] + log_mag[3k] + ...
        # This is equivalent to multiplying the spectra, which is classic HPS.
        n_bins = len(log_mag)
        hps = log_mag.copy()
        for h in range(2, self.n_harmonics + 2):
            # Downsample by h: log_mag[h], log_mag[2h], log_mag[3h], ...
            ds = log_mag[::h]
            # Add it to hps (only up to the length of ds)
            length = min(len(hps), len(ds))
            hps[:length] += ds[:length]

        # --- Search band ---
        band_hps = hps[self._k_min: self._k_max_hps + 1]
        if band_hps.size == 0:
            return None, 0.0

        # --- Confidence: peak/median of HPS in band ---
        peak_hps = float(band_hps.max())
        med_hps  = float(np.median(band_hps))
        # HPS is in log domain so we exponentiate for a linear ratio
        # exp(peak - median) = peak_linear / median_linear
        conf = float(np.exp(peak_hps - med_hps))

        # --- Best bin ---
        best_rel = int(np.argmax(band_hps))
        k_best = self._k_min + best_rel

        # --- Parabolic interpolation on the RAW magnitude for accurate frequency ---
        # (interpolate on mag, not hps, for cleaner peak shape)
        f0 = self._parabolic_peak(mag, k_best)

        if not (self.fmin <= f0 <= self.fmax):
            return None, conf

        return float(f0), conf

    def _parabolic_peak(self, mag: np.ndarray, k: int) -> float:
        """Sub-bin frequency accuracy via parabolic interpolation."""
        if k <= 0 or k >= len(mag) - 1:
            return float(k) * self.sr / self.n_fft
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-12:
            return float(k) * self.sr / self.n_fft
        delta = 0.5 * (a - c) / denom
        return (float(k) + delta) * self.sr / self.n_fft


# Aliases for backward compat
YINPitchDetector = FFTPitchDetector
HPSPitchDetector = FFTPitchDetector