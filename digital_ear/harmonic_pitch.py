# Pitch salience detector based on MELODIA (Salamon & Gomez, IEEE TASLP 2012)
#
# Spectral peaks -> harmonic summation with cos^2 spreading in 10-cent bins.
# Uses IF (instantaneous frequency) from phase diff for sub-bin accuracy,
# A-weighting to boost the melody range, 12 harmonics with alpha decay.
# Everything float32 for Pi efficiency.

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class HarmonicPitchDetector:
    """Pitch detector using MELODIA-style salience in 10-cent bins.
    Extracts spectral peaks, does harmonic summation with cos^2 spreading,
    returns candidates for the Viterbi melody tracker."""

    sr: float
    n_fft: int = 2048           # actual frame length (samples)
    hop: int = 512              # hop size for IF computation
    fmin: float = 80.0
    fmax: float = 1000.0
    n_harmonics: int = 12       # past h=12 the weight is negligible
    zero_pad: int = 4096        # FFT size after zero-padding, IF compensates
    conf_threshold: float = 3.0 # peak_score / median_score must exceed this
    subharm_ratio: float = 0.6  # prefer sub-harmonic if its score >= this fraction of peak
    alpha: float = 0.8          # harmonic weight decay: weight_h = alpha^(h-1)
    gamma: float = 1.0          # magnitude compression (unused, kept for compat)
    mag_threshold_db: float = 40.0  # peak threshold, dB below max

    # pre-computed
    _window: np.ndarray = field(init=False, repr=False)
    _bin_hz: float = field(init=False, repr=False)
    _a_weights: np.ndarray = field(init=False, repr=False)
    # cents-domain parameters
    _f_ref: float = field(init=False, repr=False)
    _cent_res: int = field(init=False, repr=False)
    _n_cent_bins: int = field(init=False, repr=False)
    _fmin_cbin: int = field(init=False, repr=False)
    _fmax_cbin: int = field(init=False, repr=False)
    # IF state
    _prev_phase: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._window = np.hanning(self.n_fft).astype(np.float32)
        self._bin_hz = self.sr / self.zero_pad

        # A-weighting for equal loudness
        n_bins = self.zero_pad // 2 + 1
        freqs = np.arange(n_bins) * self._bin_hz
        freqs[0] = 1.0  # avoid div-by-zero at DC
        self._a_weights = self._compute_a_weights(freqs)

        # cents domain: 600 bins x 10 cents = 6000 cents (55 Hz to ~1750 Hz)
        self._f_ref = 55.0
        self._cent_res = 10
        self._n_cent_bins = 600

        self._fmin_cbin = max(0, int(np.floor(
            1200.0 * np.log2(self.fmin / self._f_ref) / self._cent_res
        )))
        self._fmax_cbin = min(self._n_cent_bins, int(np.ceil(
            1200.0 * np.log2(self.fmax / self._f_ref) / self._cent_res
        )) + 1)

    @staticmethod
    def _compute_a_weights(freqs: np.ndarray) -> np.ndarray:
        """A-weighting curve: boosts 1-4 kHz, attenuates bass and high treble."""
        f2 = freqs ** 2
        num = 12194.0**2 * f2**2
        den = (
            (f2 + 20.6**2)
            * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
            * (f2 + 12194.0**2)
        )
        w = num / (den + 1e-20)
        w /= w.max() + 1e-20  # normalize peak to 1.0
        return w.astype(np.float32)

    def _extract_peaks(
        self, mag: np.ndarray, phase: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract spectral peaks with IF refinement (falls back to parabolic).
        Returns (frequencies_hz, amplitudes) as float32."""
        n_bins = mag.shape[0]
        mag_max = mag.max()
        if mag_max < 1e-20:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        threshold = mag_max * 10 ** (-self.mag_threshold_db / 20.0)

        # local maxima detection (vectorized)
        is_peak = np.zeros(n_bins, dtype=bool)
        is_peak[1:-1] = (
            (mag[1:-1] > mag[:-2]) &
            (mag[1:-1] > mag[2:]) &
            (mag[1:-1] > threshold)
        )

        peak_indices = np.where(is_peak)[0]
        if len(peak_indices) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        # parabolic interp for sub-bin accuracy (baseline before IF)
        s0 = mag[peak_indices - 1]
        s1 = mag[peak_indices]
        s2 = mag[peak_indices + 1]
        denom = 2.0 * (2.0 * s1 - s2 - s0)
        delta = np.where(np.abs(denom) > 1e-12, (s0 - s2) / denom, 0.0)

        freqs = (peak_indices + delta) * self._bin_hz
        amps = s1

        # IF refinement: phase difference gives better freq estimates
        if phase is not None and self._prev_phase is not None:
            expected_advance = 2.0 * np.pi * peak_indices * self.hop / self.zero_pad
            dp = phase[peak_indices] - self._prev_phase[peak_indices]
            dev = dp - expected_advance
            dev = (dev + np.pi) % (2.0 * np.pi) - np.pi  # unwrap
            if_freqs = (peak_indices * self._bin_hz) + dev / (2.0 * np.pi) * (self.sr / self.hop)
            # only trust IF when it's within 2 bins of the mag peak
            reasonable = np.abs(if_freqs - peak_indices * self._bin_hz) < (2.0 * self._bin_hz)
            freqs = np.where(reasonable, if_freqs, freqs)

        # Filter to relevant frequency range [f_ref, fmax * n_harmonics]
        freq_hi = min(self.fmax * self.n_harmonics, self.sr / 2)
        valid = (freqs >= self._f_ref) & (freqs <= freq_hi)

        return freqs[valid].astype(np.float32), amps[valid]

    def _compute_scores(self, frame: np.ndarray) -> tuple[np.ndarray | None, float, float]:
        """Salience in cents domain. Returns (scores, med_score, tonality) or (None, 0, 0).
        tonality: 1 = clean tone, 0 = noise/drums."""
        if frame.shape[0] < self.n_fft:
            return None, 0.0, 0.0

        xw = frame[:self.n_fft].astype(np.float32) * self._window
        padded = np.zeros(self.zero_pad, dtype=np.float32)
        padded[:self.n_fft] = xw

        X = np.fft.rfft(padded)
        mag = np.abs(X).astype(np.float32)
        phase = np.angle(X)  # keep phase for IF computation
        del X  # free complex array
        mag[0] = 0.0
        n_bins = mag.shape[0]

        # A-weight: boost melody range, suppress bass
        mag *= self._a_weights[:n_bins]

        # spectral flatness -> tonality estimate
        k_min_flat = max(1, int(np.ceil(self.fmin / self._bin_hz)))
        k_max_flat = min(n_bins, int(np.floor(self.fmax / self._bin_hz)) * min(self.n_harmonics, 10))
        mag_slice = mag[k_min_flat:k_max_flat]
        if len(mag_slice) > 0:
            mag_slice_safe = np.maximum(mag_slice, 1e-20)
            log_mean = float(np.mean(np.log(mag_slice_safe)))
            arith_mean = float(np.mean(mag_slice_safe))
            if arith_mean > 1e-20:
                flatness = float(np.exp(log_mean) / arith_mean)
            else:
                flatness = 1.0
        else:
            flatness = 1.0
        tonality = 1.0 - min(1.0, max(0.0, flatness))

        # extract spectral peaks (with IF refinement)
        peak_freqs, peak_amps = self._extract_peaks(mag, phase)
        self._prev_phase = phase  # store for next frame's IF

        if len(peak_freqs) == 0:
            return None, 0.0, tonality

        # salience in cents domain
        salience = np.zeros(self._n_cent_bins, dtype=np.float32)
        bps = 100 // self._cent_res  # bins per semitone = 10
        f_max_hz = self._f_ref * 2 ** (self._n_cent_bins * self._cent_res / 1200.0)
        half_pi = np.pi / 2.0
        offsets = np.arange(-bps, bps + 1, dtype=np.int32)  # shape (21,)

        for h in range(1, self.n_harmonics + 1):
            weight_h = self.alpha ** (h - 1)
            sub_freqs = peak_freqs / h

            # Filter to valid cents range
            valid = (sub_freqs >= self._f_ref) & (sub_freqs < f_max_hz)
            if not valid.any():
                continue

            sf = sub_freqs[valid]
            sa = peak_amps[valid] * weight_h

            # Map to cent bins (exact float position)
            cent_bins_exact = 1200.0 * np.log2(sf / self._f_ref) / self._cent_res

            # Spread contribution over ±1 semitone (±bps bins) with cos² weighting
            center_bins = np.round(cent_bins_exact).astype(np.int32)
            target_bins = center_bins[:, None] + offsets[None, :]  # (V, 21)

            deltas = np.abs(target_bins - cent_bins_exact[:, None]) / bps
            cos_w = np.where(deltas <= 1.0, np.cos(deltas * half_pi) ** 2, 0.0)

            contributions = cos_w * sa[:, None]  # (V, 21)

            # scatter-add (bincount faster than np.add.at)
            valid_mask = (target_bins >= 0) & (target_bins < self._n_cent_bins) & (contributions > 0)
            flat_targets = target_bins[valid_mask]
            flat_contribs = contributions[valid_mask]

            if len(flat_targets) > 0:
                salience += np.bincount(
                    flat_targets, weights=flat_contribs,
                    minlength=self._n_cent_bins
                )[:self._n_cent_bins]

        # pull out scores in fmin..fmax range
        n_scores = self._fmax_cbin - self._fmin_cbin
        if n_scores <= 0:
            return None, 0.0, tonality

        scores = salience[self._fmin_cbin:self._fmax_cbin].copy()
        med_score = float(np.median(scores)) + 1e-12
        return scores, med_score, tonality

    def _cbin_to_hz(self, cbin: float) -> float:
        """Convert a cent-domain bin index to Hz."""
        return self._f_ref * 2 ** (cbin * self._cent_res / 1200.0)

    def estimate(self, frame: np.ndarray) -> tuple[float | None, float]:
        """Single-best f0 estimate from a time-domain frame.
        Returns (f0_hz or None, confidence)."""
        result = self._compute_scores(frame)
        if result[0] is None:
            return None, 0.0
        scores, med_score, tonality = result
        n_candidates = scores.shape[0]

        # Find the best candidate
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        # sub-harmonic preference (f/2 is 1200 cents = 120 bins below, etc)
        for divisor in (2, 3):
            offset = int(round(1200.0 * np.log2(divisor) / self._cent_res))
            sub_idx = best_idx - offset
            if 0 <= sub_idx < n_candidates:
                if scores[sub_idx] >= self.subharm_ratio * best_score:
                    best_idx = sub_idx
                    best_score = scores[sub_idx]
                    break  # prefer lowest valid sub-harmonic

        # confidence = peak / median
        conf = best_score / med_score

        if conf < self.conf_threshold:
            return None, conf

        # parabolic interp for sub-bin accuracy
        f0_bin = self._parabolic_peak(scores, best_idx)
        f0_hz = self._cbin_to_hz(self._fmin_cbin + f0_bin)

        # clamp to search range
        if f0_hz < self.fmin or f0_hz > self.fmax:
            return None, conf

        return f0_hz, conf

    def estimate_candidates(
        self, frame: np.ndarray, n_top: int = 5, min_conf: float = 2.0
    ) -> tuple[list[tuple[float, float]], float]:
        """Top-N pitch candidates as (f0_hz, confidence) pairs, plus tonality.
        Lower confidence floor than estimate() so Viterbi can decide."""
        result = self._compute_scores(frame)
        if result[0] is None:
            return [], 0.0
        scores, med_score, tonality = result

        n_candidates = scores.shape[0]

        # local peak detection
        conf_threshold = min_conf * med_score
        if n_candidates >= 3:
            is_peak = np.zeros(n_candidates, dtype=bool)
            is_peak[1:-1] = (
                (scores[1:-1] >= scores[:-2]) &
                (scores[1:-1] >= scores[2:]) &
                (scores[1:-1] >= conf_threshold)
            )
            # check endpoints
            if scores[0] >= conf_threshold and scores[0] >= scores[1]:
                is_peak[0] = True
            if scores[-1] >= conf_threshold and scores[-1] >= scores[-2]:
                is_peak[-1] = True
            peak_indices = np.where(is_peak)[0].tolist()
        elif n_candidates > 0:
            peak_indices = [i for i in range(n_candidates) if scores[i] >= conf_threshold]
        else:
            peak_indices = []

        if not peak_indices:
            return [], tonality

        # sort by score descending
        peak_indices.sort(key=lambda i: scores[i], reverse=True)

        # no sub-harmonic preference here -- Viterbi resolves octave ambiguity via continuity
        candidates: list[tuple[float, float]] = []
        used_midi: set[int] = set()  # avoid duplicate pitches

        for pidx in peak_indices:
            if len(candidates) >= n_top:
                break

            # parabolic interp
            f0_bin = self._parabolic_peak(scores, pidx)
            f0_hz = self._cbin_to_hz(self._fmin_cbin + f0_bin)

            if f0_hz < self.fmin or f0_hz > self.fmax:
                continue

            # deduplicate by MIDI note
            midi = int(round(69 + 12 * np.log2(f0_hz / 440.0)))
            if midi in used_midi:
                continue
            used_midi.add(midi)

            conf = scores[pidx] / med_score
            candidates.append((f0_hz, conf))

        return candidates, tonality

    def _parabolic_peak(self, scores: np.ndarray, idx: int) -> float:
        """Parabolic interpolation to refine peak position."""
        if idx <= 0 or idx >= scores.shape[0] - 1:
            return float(idx)
        s0 = scores[idx - 1]
        s1 = scores[idx]
        s2 = scores[idx + 1]
        denom = 2.0 * (2.0 * s1 - s2 - s0)
        if abs(denom) < 1e-12:
            return float(idx)
        delta = (s0 - s2) / denom
        return idx + delta
