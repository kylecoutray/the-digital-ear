#!/usr/bin/env python3
"""
Compare clean vs noisy frame dumps to identify noise impact patterns.

Usage:
    python compare_frames.py test_outputs/i3_clean_frames.csv test_outputs/i3_frames.csv
"""

import csv
import sys
import math
from collections import Counter


def load_frames(path):
    frames = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                'idx': int(row['frame_idx']),
                'time': float(row['time_sec']),
                'f0': float(row['f0_hz']),
                'conf': float(row['conf']),
                'rms': float(row['rms']),
                'voiced': int(row['voiced']),
            })
    return frames


def hz_to_note(hz):
    """Convert Hz to MIDI note name."""
    if hz <= 0:
        return "---"
    midi = 69 + 12 * math.log2(hz / 440.0)
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note = int(round(midi))
    octave = note // 12 - 1
    name = names[note % 12]
    return f"{name}{octave}"


def semitone_diff(hz1, hz2):
    """Absolute semitone difference between two frequencies."""
    if hz1 <= 0 or hz2 <= 0:
        return float('inf')
    return abs(12.0 * math.log2(hz1 / hz2))


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_frames.py <clean_frames.csv> <noisy_frames.csv>")
        sys.exit(1)

    clean = load_frames(sys.argv[1])
    noisy = load_frames(sys.argv[2])

    # Align by frame count (they may differ in length due to different audio lengths)
    n = min(len(clean), len(noisy))
    print(f"Clean frames: {len(clean)}, Noisy frames: {len(noisy)}, Comparing: {n}")
    print()

    # --- Analysis categories ---
    total = 0
    pitch_match = 0          # within 1 semitone
    pitch_close = 0          # within 2 semitones
    pitch_octave = 0         # octave error (11-13 semitones)
    pitch_wrong = 0          # >2 semitones, not octave
    both_unvoiced = 0
    clean_voiced_noisy_not = 0   # missed notes
    clean_not_noisy_voiced = 0   # phantom notes

    conf_diffs = []
    rms_diffs = []

    # Time-bucketed analysis (1-second buckets)
    bucket_errors = Counter()
    bucket_total = Counter()

    # Frequency-bucketed analysis
    freq_correct = Counter()
    freq_wrong = Counter()

    # Confidence at correct vs wrong
    conf_when_correct = []
    conf_when_wrong = []

    # Octave error details
    octave_errors = []

    for i in range(n):
        c = clean[i]
        ny = noisy[i]
        total += 1

        t_bucket = int(c['time'])
        bucket_total[t_bucket] += 1

        # Voicing comparison
        if c['voiced'] == 0 and ny['voiced'] == 0:
            both_unvoiced += 1
            pitch_match += 1
            continue
        if c['voiced'] == 1 and ny['voiced'] == 0:
            clean_voiced_noisy_not += 1
            bucket_errors[t_bucket] += 1
            continue
        if c['voiced'] == 0 and ny['voiced'] == 1:
            clean_not_noisy_voiced += 1
            bucket_errors[t_bucket] += 1
            continue

        # Both voiced — compare pitch
        diff = semitone_diff(c['f0'], ny['f0'])
        conf_diffs.append(ny['conf'] - c['conf'])
        rms_diffs.append(ny['rms'] - c['rms'])

        # Frequency bucket (round to nearest note)
        note_name = hz_to_note(c['f0'])

        if diff <= 1.0:
            pitch_match += 1
            freq_correct[note_name] += 1
            conf_when_correct.append(ny['conf'])
        elif diff <= 2.0:
            pitch_close += 1
            freq_correct[note_name] += 1
            conf_when_correct.append(ny['conf'])
        elif 11.0 <= diff <= 13.0:
            pitch_octave += 1
            freq_wrong[note_name] += 1
            conf_when_wrong.append(ny['conf'])
            octave_errors.append({
                'time': c['time'],
                'clean_hz': c['f0'], 'clean_note': hz_to_note(c['f0']),
                'noisy_hz': ny['f0'], 'noisy_note': hz_to_note(ny['f0']),
                'noisy_conf': ny['conf'],
            })
            bucket_errors[t_bucket] += 1
        else:
            pitch_wrong += 1
            freq_wrong[note_name] += 1
            conf_when_wrong.append(ny['conf'])
            bucket_errors[t_bucket] += 1

    # --- Print report ---
    print("=" * 60)
    print("FRAME-BY-FRAME COMPARISON")
    print("=" * 60)
    print(f"Total compared:       {total}")
    print(f"Pitch match (≤1st):   {pitch_match}  ({100*pitch_match/total:.1f}%)")
    print(f"Pitch close (≤2st):   {pitch_close}  ({100*pitch_close/total:.1f}%)")
    print(f"Octave errors:        {pitch_octave}  ({100*pitch_octave/total:.1f}%)")
    print(f"Wrong pitch (>2st):   {pitch_wrong}  ({100*pitch_wrong/total:.1f}%)")
    print(f"Both unvoiced:        {both_unvoiced}  ({100*both_unvoiced/total:.1f}%)")
    print(f"Missed (clean→unv):   {clean_voiced_noisy_not}  ({100*clean_voiced_noisy_not/total:.1f}%)")
    print(f"Phantom (unv→noisy):  {clean_not_noisy_voiced}  ({100*clean_not_noisy_voiced/total:.1f}%)")

    accuracy = (pitch_match + pitch_close) / total * 100
    print(f"\nOverall frame accuracy (≤2st): {accuracy:.1f}%")

    # Confidence analysis
    if conf_diffs:
        avg_conf_diff = sum(conf_diffs) / len(conf_diffs)
        print(f"\nAvg confidence change (noisy-clean): {avg_conf_diff:+.2f}")
    if conf_when_correct:
        print(f"Avg noisy confidence when CORRECT: {sum(conf_when_correct)/len(conf_when_correct):.2f}")
    if conf_when_wrong:
        print(f"Avg noisy confidence when WRONG:   {sum(conf_when_wrong)/len(conf_when_wrong):.2f}")

    # Time-bucketed error rate
    print(f"\n{'='*60}")
    print("ERROR RATE BY TIME (seconds with >20% error)")
    print("=" * 60)
    problem_seconds = []
    for t in sorted(bucket_total.keys()):
        err = bucket_errors.get(t, 0)
        tot = bucket_total[t]
        rate = err / tot * 100
        if rate > 20:
            problem_seconds.append(t)
            print(f"  t={t:3d}s: {err:3d}/{tot:3d} errors ({rate:.0f}%)")

    if not problem_seconds:
        print("  No seconds with >20% error rate")

    # Frequency analysis — which notes are hardest?
    print(f"\n{'='*60}")
    print("ACCURACY BY NOTE (notes with >10 total frames)")
    print("=" * 60)
    all_notes = set(freq_correct.keys()) | set(freq_wrong.keys())
    note_stats = []
    for note in all_notes:
        correct = freq_correct.get(note, 0)
        wrong = freq_wrong.get(note, 0)
        total_note = correct + wrong
        if total_note > 10:
            note_stats.append((note, correct, wrong, total_note))

    note_stats.sort(key=lambda x: x[2] / x[3], reverse=True)
    for note, correct, wrong, total_note in note_stats:
        acc = correct / total_note * 100
        print(f"  {note:4s}: {correct:4d} correct, {wrong:3d} wrong  ({acc:.0f}% accuracy)")

    # Octave errors detail (first 20)
    if octave_errors:
        print(f"\n{'='*60}")
        print(f"OCTAVE ERRORS (showing first 20 of {len(octave_errors)})")
        print("=" * 60)
        for oe in octave_errors[:20]:
            direction = "UP" if oe['noisy_hz'] > oe['clean_hz'] else "DOWN"
            print(f"  t={oe['time']:7.2f}s  clean={oe['clean_note']:4s}({oe['clean_hz']:6.1f}Hz) "
                  f"→ noisy={oe['noisy_note']:4s}({oe['noisy_hz']:6.1f}Hz)  "
                  f"[{direction}]  conf={oe['noisy_conf']:.1f}")

    # Summary of actionable patterns
    print(f"\n{'='*60}")
    print("ACTIONABLE INSIGHTS")
    print("=" * 60)

    if pitch_octave > 0:
        pct = 100 * pitch_octave / total
        print(f"• Octave errors: {pct:.1f}% — causal octave correction may need tuning")

    if clean_not_noisy_voiced > 0:
        pct = 100 * clean_not_noisy_voiced / total
        print(f"• Phantom voicing: {pct:.1f}% — noise creates false pitch detections")

    if clean_voiced_noisy_not > 0:
        pct = 100 * clean_voiced_noisy_not / total
        print(f"• Missed voicing: {pct:.1f}% — noise masks real pitch content")

    if conf_when_wrong:
        avg_w = sum(conf_when_wrong) / len(conf_when_wrong)
        if conf_when_correct:
            avg_c = sum(conf_when_correct) / len(conf_when_correct)
            if avg_w < avg_c * 0.7:
                print(f"• Confidence separates correct/wrong well ({avg_c:.1f} vs {avg_w:.1f})")
                print(f"  → A confidence threshold of ~{(avg_c+avg_w)/2:.1f} could filter errors")
            else:
                print(f"• Confidence does NOT separate well ({avg_c:.1f} vs {avg_w:.1f})")
                print(f"  → Need spectral/harmonic features, not just confidence")

    if problem_seconds:
        print(f"• Problem time regions: {problem_seconds[:10]}s")
        print(f"  → Use --dual to listen to these specific moments")


if __name__ == '__main__':
    main()
