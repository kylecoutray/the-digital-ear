# Online Viterbi melody extractor
#
# One frame at a time via push(), melody decisions come out with ~50 frame
# lag (~580ms). Ring buffer of backpointers for online backtrace.
# Tonality/density smoothing is all causal. ~23 KB constant memory.
#
# Reference: online Viterbi / fixed-lag smoothing for HMMs.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class MelodyExtractor:
    """Online Viterbi melody extractor. Feed frames via push(), get melody
    decisions back delayed by `lag` frames. Call flush() at end of stream."""

    hop_sec: float
    lag: int = 50                        # fixed-lag backtrace depth (~580 ms)
    jump_weight_tonal: float = 1.2
    jump_weight_noisy: float = 0.5
    voicing_cost_tonal: float = 4.0
    voicing_cost_noisy: float = 7.0
    tonality_emission_scale: float = 0.5
    density_emission_boost: float = 1.5
    density_voicing_floor: float = 0.4
    density_jump_floor: float = 0.7
    octave_smooth_sec: float = 5.0
    # prev frame's smoothed values (for transition averaging)
    _prev_ton: float = field(init=False, repr=False, default=0.5)
    _prev_den: float = field(init=False, repr=False, default=1.0)

    # internal state
    _V_prev: np.ndarray | None = field(init=False, repr=False, default=None)
    _states_prev: list = field(init=False, repr=False, default_factory=list)
    _ring: list = field(init=False, repr=False, default_factory=list)
    _frame_idx: int = field(init=False, repr=False, default=0)

    # smoothing buffers
    _tonality_buf: deque = field(init=False, repr=False)
    _density_buf: deque = field(init=False, repr=False)
    _recent_voiced: deque = field(init=False, repr=False)

    # octave correction
    _octave_ring: deque = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ton_window = max(1, int(0.5 / self.hop_sec))  # 0.5s moving avg
        self._tonality_buf = deque(maxlen=ton_window)
        self._density_buf = deque(maxlen=ton_window)

        density_window = max(1, int(10.0 / self.hop_sec))  # 10s lookback
        self._recent_voiced = deque(maxlen=density_window)

        oct_window = max(1, int(self.octave_smooth_sec / self.hop_sec))
        self._octave_ring = deque(maxlen=oct_window)

    def push(
        self,
        candidates: list[tuple[float, float]],
        tonality: float,
    ) -> float | None:
        """Process one frame. Returns f0_hz (or None) for the frame
        emitted `lag` frames behind current, or None if still filling up."""

        # density estimation
        has_cands = 1.0 if len(candidates) > 0 else 0.0
        self._recent_voiced.append(has_cands)
        density = sum(self._recent_voiced) / len(self._recent_voiced)

        # smoothing
        self._tonality_buf.append(tonality)
        self._density_buf.append(density)
        ton_smooth = sum(self._tonality_buf) / len(self._tonality_buf)
        den_smooth = sum(self._density_buf) / len(self._density_buf)

        # frame states: candidates + unvoiced
        frame_states = [(hz, conf) for hz, conf in candidates]
        frame_states.append((None, 0.0))  # unvoiced
        ns_cur = len(frame_states)

        # Viterbi step
        if self._V_prev is None:
            V_cur = np.zeros(ns_cur, dtype=np.float32)
            for j, (hz_j, conf) in enumerate(frame_states):
                V_cur[j] = self._emission(conf, ton_smooth, den_smooth)
            bp = np.zeros(ns_cur, dtype=np.int32)
        else:
            V_cur = np.full(ns_cur, -1e18, dtype=np.float32)
            bp = np.zeros(ns_cur, dtype=np.int32)

            ton_avg = 0.5 * (ton_smooth + self._prev_ton)
            den_avg = 0.5 * (den_smooth + self._prev_den)

            for j in range(ns_cur):
                hz_j = frame_states[j][0]
                emit_j = self._emission(frame_states[j][1], ton_smooth, den_smooth)
                best_val = -1e18
                best_i = 0

                for i in range(len(self._states_prev)):
                    hz_i = self._states_prev[i][0]
                    trans = self._transition(hz_i, hz_j, ton_avg, den_avg)
                    val = self._V_prev[i] + trans
                    if val > best_val:
                        best_val = val
                        best_i = i

                V_cur[j] = best_val + emit_j
                bp[j] = best_i

        # store in ring buffer
        self._ring.append((bp, frame_states, V_cur.copy()))

        self._V_prev = V_cur
        self._states_prev = frame_states
        self._prev_ton = ton_smooth
        self._prev_den = den_smooth
        self._frame_idx += 1

        # emit oldest via backtrace once ring exceeds lag
        if len(self._ring) > self.lag:
            return self._emit_oldest()
        return None

    def flush(self) -> list[float | None]:
        """Emit remaining lagged frames at end of stream."""
        results: list[float | None] = []
        while len(self._ring) > 0:
            results.append(self._emit_oldest())
        return results

    def _emit_oldest(self) -> float | None:
        """Backtrace from current best to oldest frame, pop it off."""
        if not self._ring:
            return None

        # current best state
        _, _, V_latest = self._ring[-1]
        state_idx = int(np.argmax(V_latest))

        # backtrace
        for i in range(len(self._ring) - 1, 0, -1):
            bp_i, _, _ = self._ring[i]
            state_idx = int(bp_i[state_idx])

        # oldest entry's decision
        _, oldest_states, _ = self._ring[0]
        hz, conf = oldest_states[state_idx]

        # octave correction
        hz = self._causal_octave_correct(hz)

        # drop oldest
        self._ring.pop(0)

        return hz

    def _causal_octave_correct(self, hz: float | None) -> float | None:
        """Fix octave jumps using 5s lookback median."""
        if hz is None or hz <= 0:
            self._octave_ring.append(0.0)
            return hz

        self._octave_ring.append(hz)

        # need enough context
        voiced = [v for v in self._octave_ring if v > 0]
        if len(voiced) < 5:
            return hz

        log2_55 = math.log2(55.0)
        cents_cur = 1200.0 * (math.log2(hz) - log2_55)
        cents_all = [1200.0 * (math.log2(v) - log2_55) for v in voiced]
        median_cents = sorted(cents_all)[len(cents_all) // 2]

        diff = cents_cur - median_cents

        # Snap down octave
        if 1150 <= diff <= 1250:
            return hz / 2.0
        # Snap down octave+fifth
        if 1850 <= diff <= 1950:
            return hz / 3.0
        # Snap up octave
        if -1250 <= diff <= -1150:
            return hz * 2.0

        return hz

    def _emission(self, conf: float, tonality: float, density: float = 1.0) -> float:
        """Emission score, weighted by tonality and density."""
        if conf <= 0:
            return 0.0
        base = math.log1p(conf)
        scale = self.tonality_emission_scale * tonality + (1.0 - self.tonality_emission_scale)
        density_boost = 1.0 + self.density_emission_boost * (1.0 - density)
        return base * scale * density_boost

    def _transition(
        self, hz_prev: float | None, hz_cur: float | None,
        tonality: float, density: float = 1.0
    ) -> float:
        """Transition cost, scales with tonality and voicing density."""
        jump_w = (
            tonality * self.jump_weight_tonal
            + (1.0 - tonality) * self.jump_weight_noisy
        )
        voice_c = (
            tonality * self.voicing_cost_tonal
            + (1.0 - tonality) * self.voicing_cost_noisy
        )

        voice_density_mult = self.density_voicing_floor + (1.0 - self.density_voicing_floor) * density
        jump_density_mult = self.density_jump_floor + (1.0 - self.density_jump_floor) * density

        voice_c *= voice_density_mult
        jump_w *= jump_density_mult

        if hz_prev is None and hz_cur is None:
            return 0.0
        if hz_prev is None or hz_cur is None:
            return -voice_c

        semitones = abs(12.0 * math.log2(hz_cur / hz_prev))
        return -jump_w * semitones
