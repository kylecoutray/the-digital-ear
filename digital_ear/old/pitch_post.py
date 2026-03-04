# FIXED pitch_post.py
#
# BUG: snap_harmonic() was CAUSING octave errors, not fixing them.
#
# The original logic: "if the new f0 is near a harmonic of prev_f0, snap to that harmonic"
# This means: if prev=200 Hz and new detection=400 Hz (a real octave jump),
# snap_harmonic sees ratio=2.0 within tolerance and SNAPS BACK to 200 Hz.
# => Genuine melodic leaps get silently suppressed.
# => The melody sounds "stuck" and wrong notes are held too long.
#
# FIX STRATEGY:
#   - Only snap DOWN (to the fundamental), never snap UP to harmonics.
#   - Only snap if the deviation is tiny (< 0.4 semitones) — pure micro-correction.
#   - Never snap across octave boundaries — those are real pitch changes.
#
# This makes the function a gentle pitch-smoothing correction rather than
# an aggressive octave-locking mechanism.

from __future__ import annotations
import math


def snap_harmonic(f0: float, prev_f0: float | None, tol_semitones: float = 0.4) -> float:
    """
    Gentle micro-correction: if f0 is almost identical to prev_f0 (within
    tol_semitones), snap to prev_f0 to reduce jitter on held notes.

    We deliberately do NOT snap to octave multiples/divisors — those are
    real pitch changes and should be left alone.

    tol_semitones: only snap if within this many semitones of prev.
                   Default 0.4 semitones (~half a MIDI step).
    """
    if prev_f0 is None or f0 <= 0.0 or prev_f0 <= 0.0:
        return f0

    err_semitones = abs(12.0 * math.log2(f0 / prev_f0))

    if err_semitones <= tol_semitones:
        # Tiny deviation — smooth it to the previous value
        return prev_f0

    # Large deviation = real pitch change. Leave it alone.
    return f0
