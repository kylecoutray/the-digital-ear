#!/usr/bin/env python3
"""
Analyze per-frame dump CSV to diagnose extraction issues.

Reads the CSV from main.py --dump-frames and shows:
  1. Confidence distribution — is the threshold cutting good candidates?
  2. RMS distribution — is the voicing gate too high/low?
  3. Pitch stability — how much does pitch flicker frame-to-frame?
  4. Voicing rate over time — are there gaps where it drops out?
  5. Pitch range histogram — where are we detecting notes?
  6. Confidence vs correctness — are high-confidence notes accurate?

Usage:
    python diagnose_frames.py test_outputs/i3_frames.csv
    python diagnose_frames.py test_outputs/i3_frames.csv --gt test_outputs/i3_clean_basic_pitch.mid
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter


def hz_to_note_name(hz: float) -> str:
    if hz <= 0:
        return "---"
    midi = int(round(69 + 12 * math.log2(hz / 440.0)))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return -1
    return int(round(69 + 12 * math.log2(hz / 440.0)))


def main():
    p = argparse.ArgumentParser(description="Diagnose per-frame extraction data")
    p.add_argument("csv_path", help="Path to frame dump CSV from --dump-frames")
    p.add_argument("--gt", help="Optional ground truth MIDI for error analysis")
    args = p.parse_args()

    # Load frames
    frames = []
    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                "idx": int(row["frame_idx"]),
                "time": float(row["time_sec"]),
                "f0": float(row["f0_hz"]),
                "conf": float(row["conf"]),
                "rms": float(row["rms"]),
                "voiced": int(row["voiced"]),
            })

    n = len(frames)
    if n == 0:
        print("No frames found!")
        return

    duration = frames[-1]["time"]
    hop_sec = frames[1]["time"] - frames[0]["time"] if n > 1 else 0.0116

    print(f"\n{'=' * 60}")
    print(f"  Frame Diagnostic Report")
    print(f"  {n} frames, {duration:.1f}s, hop={hop_sec * 1000:.1f}ms")
    print(f"{'=' * 60}\n")

    # ── 1. Voicing stats ──
    voiced = [f for f in frames if f["voiced"]]
    unvoiced = [f for f in frames if not f["voiced"]]
    print(f"── Voicing ──")
    print(f"  Voiced:   {len(voiced):>6} ({100 * len(voiced) / n:.1f}%)")
    print(f"  Unvoiced: {len(unvoiced):>6} ({100 * len(unvoiced) / n:.1f}%)")

    # Voicing gaps (consecutive unvoiced runs)
    if unvoiced:
        gaps = []
        gap_start = None
        for f in frames:
            if not f["voiced"]:
                if gap_start is None:
                    gap_start = f["time"]
            else:
                if gap_start is not None:
                    gaps.append(f["time"] - gap_start)
                    gap_start = None
        if gap_start is not None:
            gaps.append(duration - gap_start)
        if gaps:
            print(f"  Unvoiced gaps: {len(gaps)} gaps, "
                  f"avg {sum(gaps) / len(gaps):.3f}s, max {max(gaps):.3f}s")
    print()

    # ── 2. Confidence distribution ──
    confs = [f["conf"] for f in voiced]
    if confs:
        confs_sorted = sorted(confs)
        print(f"── Confidence (voiced frames only) ──")
        print(f"  Min:    {confs_sorted[0]:.2f}")
        print(f"  25th:   {confs_sorted[len(confs) // 4]:.2f}")
        print(f"  Median: {confs_sorted[len(confs) // 2]:.2f}")
        print(f"  75th:   {confs_sorted[3 * len(confs) // 4]:.2f}")
        print(f"  Max:    {confs_sorted[-1]:.2f}")
        print(f"  Mean:   {sum(confs) / len(confs):.2f}")

        # Histogram
        brackets = [(0, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 50)]
        print(f"\n  Confidence histogram:")
        for lo, hi in brackets:
            count = sum(1 for c in confs if lo <= c < hi)
            bar = "█" * (count * 40 // len(confs))
            print(f"    [{lo:>4.0f}-{hi:>4.0f}) {count:>5} {100 * count / len(confs):>5.1f}% {bar}")
        print()

    # ── 3. RMS distribution ──
    rms_vals = [f["rms"] for f in frames]
    rms_sorted = sorted(rms_vals)
    print(f"── RMS Energy ──")
    print(f"  Min:    {rms_sorted[0]:.6f}")
    print(f"  25th:   {rms_sorted[len(rms_sorted) // 4]:.6f}")
    print(f"  Median: {rms_sorted[len(rms_sorted) // 2]:.6f}")
    print(f"  75th:   {rms_sorted[3 * len(rms_sorted) // 4]:.6f}")
    print(f"  Max:    {rms_sorted[-1]:.6f}")

    # Frames below common thresholds
    for th in [0.001, 0.003, 0.005, 0.01]:
        below = sum(1 for r in rms_vals if r < th)
        print(f"  Below {th}: {below} frames ({100 * below / n:.1f}%)")
    print()

    # ── 4. Pitch stability ──
    voiced_f0 = [(f["time"], f["f0"]) for f in voiced if f["f0"] > 0]
    if len(voiced_f0) > 1:
        jumps = []
        octave_jumps = 0
        for i in range(1, len(voiced_f0)):
            t_prev, f0_prev = voiced_f0[i - 1]
            t_cur, f0_cur = voiced_f0[i]
            if t_cur - t_prev > 0.05:  # skip gaps
                continue
            semitones = abs(12 * math.log2(f0_cur / f0_prev))
            jumps.append(semitones)
            if 11 <= semitones <= 13 or 18 <= semitones <= 20:
                octave_jumps += 1

        if jumps:
            jumps_sorted = sorted(jumps)
            print(f"── Pitch Stability (frame-to-frame) ──")
            print(f"  Median jump:  {jumps_sorted[len(jumps) // 2]:.2f} semitones")
            print(f"  95th pct:     {jumps_sorted[int(0.95 * len(jumps))]:.2f} semitones")
            print(f"  Max jump:     {jumps_sorted[-1]:.2f} semitones")
            print(f"  Octave jumps: {octave_jumps} ({100 * octave_jumps / len(jumps):.1f}%)")

            # Jump size histogram
            jump_brackets = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 12), (12, 50)]
            print(f"\n  Jump histogram:")
            for lo, hi in jump_brackets:
                count = sum(1 for j in jumps if lo <= j < hi)
                bar = "█" * (count * 40 // len(jumps))
                print(f"    [{lo:>4.1f}-{hi:>4.1f}) {count:>5} {100 * count / len(jumps):>5.1f}% {bar}")
            print()

    # ── 5. Pitch range ──
    if voiced_f0:
        midi_notes = [hz_to_midi(f0) for _, f0 in voiced_f0]
        note_counts = Counter(midi_notes)
        most_common = note_counts.most_common(10)

        print(f"── Pitch Range ──")
        min_hz = min(f0 for _, f0 in voiced_f0)
        max_hz = max(f0 for _, f0 in voiced_f0)
        print(f"  Range: {min_hz:.1f} Hz ({hz_to_note_name(min_hz)}) → "
              f"{max_hz:.1f} Hz ({hz_to_note_name(max_hz)})")
        print(f"  Unique MIDI notes: {len(note_counts)}")
        print(f"\n  Top 10 most frequent notes:")
        for midi, count in most_common:
            hz = 440.0 * 2 ** ((midi - 69) / 12.0)
            pct = 100 * count / len(midi_notes)
            bar = "█" * int(pct)
            print(f"    {hz_to_note_name(hz):>4} ({hz:>6.1f} Hz)  {count:>5} frames  {pct:>5.1f}% {bar}")
        print()

    # ── 6. Voicing rate over time (10s windows) ──
    print(f"── Voicing Rate Over Time (10s windows) ──")
    window = 10.0
    t = 0.0
    while t < duration:
        t_end = min(t + window, duration)
        window_frames = [f for f in frames if t <= f["time"] < t_end]
        if window_frames:
            v_rate = sum(1 for f in window_frames if f["voiced"]) / len(window_frames)
            avg_conf = sum(f["conf"] for f in window_frames if f["voiced"]) / max(1, sum(1 for f in window_frames if f["voiced"]))
            avg_rms = sum(f["rms"] for f in window_frames) / len(window_frames)
            bar = "█" * int(v_rate * 30)
            print(f"  {t:>5.0f}-{t_end:>5.0f}s  voiced={v_rate:>5.1%}  "
                  f"conf={avg_conf:>5.1f}  rms={avg_rms:.4f}  {bar}")
        t += window
    print()

    # ── 7. Actionable recommendations ──
    print(f"── Recommendations ──")

    if voiced and confs:
        median_conf = sorted(confs)[len(confs) // 2]
        if median_conf < 5.0:
            print(f"  ⚠ Median confidence is low ({median_conf:.1f}). Consider:")
            print(f"    - Lowering --conf-th (currently filtering marginal candidates)")
            print(f"    - The signal may be too noisy for reliable pitch detection")
        elif median_conf > 15.0:
            print(f"  ✓ Confidence is high ({median_conf:.1f}). Threshold is fine.")
        else:
            print(f"  ✓ Confidence is moderate ({median_conf:.1f}).")

    voiced_pct = 100 * len(voiced) / n
    if voiced_pct > 95:
        print(f"  ⚠ Voicing rate is {voiced_pct:.0f}% — almost everything is voiced.")
        print(f"    - If there are silent sections, --rms-th may be too low")
        print(f"    - Or --conf-th is too low (accepting noise as pitch)")
    elif voiced_pct < 30:
        print(f"  ⚠ Voicing rate is only {voiced_pct:.0f}% — most frames are unvoiced.")
        print(f"    - --rms-th may be too high (gating real signal)")
        print(f"    - Or --conf-th is too high (rejecting good candidates)")

    if jumps:
        big_jumps = sum(1 for j in jumps if j > 5)
        big_pct = 100 * big_jumps / len(jumps)
        if big_pct > 5:
            print(f"  ⚠ {big_pct:.1f}% of frames have >5 semitone jumps.")
            print(f"    - Viterbi transition costs may be too low")
            print(f"    - Or HPSS isn't filtering enough percussion")
        if octave_jumps > len(jumps) * 0.02:
            print(f"  ⚠ {octave_jumps} octave jumps detected ({100 * octave_jumps / len(jumps):.1f}%).")
            print(f"    - Octave correction may need wider lookback window")
            print(f"    - Or harmonic salience is favoring overtones")

    below_003 = sum(1 for r in rms_vals if r < 0.003)
    if below_003 == 0:
        print(f"  ℹ No frames below RMS 0.003 — signal is always loud enough")
        print(f"    - RMS gate is not filtering anything at current threshold")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
