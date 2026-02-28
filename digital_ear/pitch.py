# iteration 4: MVP CORE -> monophonic pitch pick using HPS (with sine-safe fallback)
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _hz_to_bin(hz: float, sr: float, n_fft: int) -> int:
    return int(np.round(hz * n_fft / sr))


@dataclass
class HPSPitchDetector:
    """
    Harmonic Product Spectrum (HPS) monophonic pitch detector.

    Works best on harmonic sources (voice, many instruments).
    For nearly-sinusoidal signals (pure tones), HPS can select subharmonics.
    This implementation detects that case and falls back to simple spectral peak picking.
    """
    sr: float
    n_fft: int
    fmin: float = 80.0
    fmax: float = 1000.0
    hps_factors: tuple[int, ...] = (2, 3, 4, 5)
    mag_floor: float = 1e-12
    conf_threshold: float = 3.0           # voiced/unvoiced gate
    harmonicity_threshold: float = 0.10   # below this => treat as near-sine, use spectral peak

    def estimate(self, mag: np.ndarray) -> tuple[float | None, float]:
        if mag.size == 0:
            return None, 0.0

        # Safe magnitude, float32, avoid log(0)
        m = mag.astype(np.float32, copy=False)
        m = np.maximum(m, self.mag_floor)

        # Zero out DC
        m0 = m.copy()
        m0[0] = self.mag_floor

        # Bin band
        k_min = max(1, _hz_to_bin(self.fmin, self.sr, self.n_fft))
        k_max = min(m0.shape[0] - 1, _hz_to_bin(self.fmax, self.sr, self.n_fft))
        if k_max <= k_min:
            return None, 0.0

        # Clamp so ALL HPS factors are valid for every k in band
        max_factor = max(self.hps_factors) if self.hps_factors else 1
        k_max_eff = min(k_max, (m0.shape[0] - 1) // max_factor)
        if k_max_eff <= k_min:
            return None, 0.0

        band_mag = m0[k_min:k_max_eff + 1]
        if band_mag.size == 0:
            return None, 0.0

        # --- Spectral peak candidate (sine-safe fallback and also useful baseline)
        rel_spec = int(np.argmax(band_mag))
        k_spec = k_min + rel_spec
        f_spec = (k_spec * self.sr) / self.n_fft

        # Harmonicity ratio: how much energy sits at harmonics of the spectral peak
        denom = float(m0[k_spec]) + 1e-12
        harm_sum = 0.0
        for h in self.hps_factors:
            kh = h * k_spec
            if kh <= (m0.shape[0] - 1):
                harm_sum += float(m0[kh])
        harmonicity = harm_sum / denom  # near 0 for pure sine

        # --- HPS score (log domain)
        base = np.log(m0)
        hps_log = base.copy()
        for f in self.hps_factors:
            ds = base[::f]
            # ds has at least k_max_eff+1 samples because of the clamp above
            hps_log[:ds.shape[0]] += ds

        band_hps = hps_log[k_min:k_max_eff + 1]
        rel_hps = int(np.argmax(band_hps))
        k_hps = k_min + rel_hps
        f_hps = (k_hps * self.sr) / self.n_fft

        # Choose between HPS and spectral peak
        # If harmonicity is low, HPS is unreliable (pure tones), so use spectral peak.
        if harmonicity < self.harmonicity_threshold:
            f0 = f_spec
            # Confidence for spectral peak: peak/median in band
            best_mag = float(m0[k_spec])
        else:
            f0 = f_hps
            best_mag = float(m0[k_hps])

        med_mag = float(np.median(band_mag)) + 1e-12
        conf = best_mag / med_mag

        if conf < self.conf_threshold:
            return None, conf

        return f0, conf