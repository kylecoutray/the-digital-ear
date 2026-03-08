#!/usr/bin/env python3
"""
Evaluate MIDI extraction accuracy against a ground truth.

Three-tier comparison:
  1. Ground truth  = Basic Pitch on CLEAN audio (the answer key)
  2. Basic Pitch   = Basic Pitch on NOISY audio (their score)
  3. Ours          = Our pipeline on NOISY audio (our score)

Both (2) and (3) are compared against (1). Whoever matches better wins.

Usage:
    # Basic: compare our output against ground truth
    python evaluate.py \
        --gt   ground_truth_clean.mid \
        --ours our_output.mid

    # Full three-way: also compare Basic Pitch on noisy
    python evaluate.py \
        --gt   ground_truth_clean.mid \
        --bp   basic_pitch_noisy.mid \
        --ours our_output.mid

    # With auto-alignment (cross-correlate audio to find time offset)
    python evaluate.py \
        --gt   ground_truth_clean.mid \
        --ours our_output.mid \
        --clean-audio clean_song.wav \
        --noisy-audio "Input 1.m4a"

    # With manual offset (ground truth starts 45.2s into clean song)
    python evaluate.py \
        --gt   ground_truth_clean.mid \
        --ours our_output.mid \
        --offset 45.2

    # Filter ground truth to melody only (highest note per timestep)
    python evaluate.py \
        --gt ground_truth_clean.mid \
        --ours our_output.mid \
        --melody-only
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
from dataclasses import dataclass


# ─── Minimal MIDI parser (no dependencies) ───────────────────────────────────

@dataclass
class MidiNote:
    note: int           # MIDI note number 0-127
    start_sec: float    # onset time in seconds
    end_sec: float      # offset time in seconds
    velocity: int       # MIDI velocity


def _read_var_len(data: bytes, offset: int) -> tuple[int, int]:
    """Read a MIDI variable-length quantity. Returns (value, new_offset)."""
    value = 0
    while True:
        b = data[offset]
        offset += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, offset


def parse_midi(path: str) -> list[MidiNote]:
    """Parse a standard MIDI file and return note events with times in seconds."""
    with open(path, "rb") as f:
        data = f.read()

    # --- MThd header ---
    if data[:4] != b"MThd":
        raise ValueError(f"Not a MIDI file: {path}")

    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, n_tracks, ticks_per_beat = struct.unpack(">HHH", data[8:14])

    # Default tempo: 120 BPM = 500000 us/beat
    us_per_beat = 500000
    sec_per_tick = us_per_beat / (ticks_per_beat * 1_000_000)

    # Tempo map: list of (abs_tick, sec_per_tick)
    tempo_map: list[tuple[int, float]] = [(0, sec_per_tick)]

    offset = 8 + header_len

    all_notes: list[MidiNote] = []

    for _track in range(n_tracks):
        if data[offset:offset + 4] != b"MTrk":
            raise ValueError(f"Expected MTrk at offset {offset}")
        track_len = struct.unpack(">I", data[offset + 4:offset + 8])[0]
        track_start = offset + 8
        track_end = track_start + track_len
        offset = track_end

        # First pass: collect tempo changes
        pos = track_start
        abs_tick = 0
        running_status = 0
        while pos < track_end:
            delta, pos = _read_var_len(data, pos)
            abs_tick += delta

            byte = data[pos]
            if byte == 0xFF:  # Meta event
                pos += 1
                meta_type = data[pos]
                pos += 1
                meta_len, pos = _read_var_len(data, pos)
                if meta_type == 0x51 and meta_len == 3:  # Tempo
                    us = (data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2]
                    spt = us / (ticks_per_beat * 1_000_000)
                    tempo_map.append((abs_tick, spt))
                pos += meta_len
            elif byte == 0xF0 or byte == 0xF7:  # SysEx
                pos += 1
                sysex_len, pos = _read_var_len(data, pos)
                pos += sysex_len
            else:
                if byte & 0x80:
                    running_status = byte
                    pos += 1
                status = running_status
                ch = status & 0x0F
                msg_type = status & 0xF0

                if msg_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    pos += 2  # 2 data bytes
                elif msg_type in (0xC0, 0xD0):
                    pos += 1  # 1 data byte
                else:
                    pos += 1  # unknown, skip

        # Sort tempo map
        tempo_map.sort(key=lambda x: x[0])

        # Helper: convert tick to seconds using tempo map
        def tick_to_sec(tick: int) -> float:
            elapsed = 0.0
            prev_tick = 0
            prev_spt = tempo_map[0][1]
            for t_tick, t_spt in tempo_map[1:]:
                if tick <= t_tick:
                    break
                elapsed += (t_tick - prev_tick) * prev_spt
                prev_tick = t_tick
                prev_spt = t_spt
            else:
                t_tick = tick  # no more tempo changes
            elapsed += (tick - prev_tick) * prev_spt
            return elapsed

        # Second pass: collect note on/off events
        pos = track_start
        abs_tick = 0
        running_status = 0
        pending: dict[tuple[int, int], tuple[int, int]] = {}  # (ch, note) -> (tick, vel)

        while pos < track_end:
            delta, pos = _read_var_len(data, pos)
            abs_tick += delta

            byte = data[pos]
            if byte == 0xFF:  # Meta
                pos += 1
                pos += 1  # meta type
                meta_len, pos = _read_var_len(data, pos)
                pos += meta_len
            elif byte == 0xF0 or byte == 0xF7:  # SysEx
                pos += 1
                sysex_len, pos = _read_var_len(data, pos)
                pos += sysex_len
            else:
                if byte & 0x80:
                    running_status = byte
                    pos += 1
                status = running_status
                ch = status & 0x0F
                msg_type = status & 0xF0

                if msg_type == 0x90:  # Note On
                    note_num = data[pos]
                    vel = data[pos + 1]
                    pos += 2
                    if vel == 0:  # Note On with vel 0 = Note Off
                        key = (ch, note_num)
                        if key in pending:
                            on_tick, on_vel = pending.pop(key)
                            all_notes.append(MidiNote(
                                note=note_num,
                                start_sec=tick_to_sec(on_tick),
                                end_sec=tick_to_sec(abs_tick),
                                velocity=on_vel,
                            ))
                    else:
                        pending[(ch, note_num)] = (abs_tick, vel)

                elif msg_type == 0x80:  # Note Off
                    note_num = data[pos]
                    pos += 2  # note + velocity
                    key = (ch, note_num)
                    if key in pending:
                        on_tick, on_vel = pending.pop(key)
                        all_notes.append(MidiNote(
                            note=note_num,
                            start_sec=tick_to_sec(on_tick),
                            end_sec=tick_to_sec(abs_tick),
                            velocity=on_vel,
                        ))

                elif msg_type in (0xA0, 0xB0, 0xE0):
                    pos += 2
                elif msg_type in (0xC0, 0xD0):
                    pos += 1
                else:
                    pos += 1

    all_notes.sort(key=lambda n: n.start_sec)
    return all_notes


# ─── Audio alignment via cross-correlation ────────────────────────────────────

def _decode_audio_mono(path: str, sr: int = 16000) -> "np.ndarray":
    """Decode any audio file to mono float32 via ffmpeg."""
    import numpy as np
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sr),
        "-v", "quiet", "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def compute_offset(clean_audio_path: str, noisy_audio_path: str,
                   sr: int = 16000) -> float:
    """Find the time offset between clean and noisy audio via cross-correlation.

    Returns offset in seconds: positive means noisy recording starts
    `offset` seconds into the clean song.
    """
    import numpy as np
    from scipy.signal import fftconvolve

    print("Loading audio for alignment...")
    clean = _decode_audio_mono(clean_audio_path, sr)
    noisy = _decode_audio_mono(noisy_audio_path, sr)

    # Use first 60s of each to save memory
    max_samples = sr * 60
    clean_chunk = clean[:max_samples]
    noisy_chunk = noisy[:max_samples]

    print("Cross-correlating...")
    corr = fftconvolve(noisy_chunk, clean_chunk[::-1], mode="full")
    lag = int(np.argmax(corr)) - len(clean_chunk) + 1
    offset_sec = lag / sr

    print(f"Alignment offset: {offset_sec:+.3f}s "
          f"(noisy starts {abs(offset_sec):.1f}s "
          f"{'into' if offset_sec > 0 else 'before'} clean)")
    return offset_sec


# ─── Filtering ────────────────────────────────────────────────────────────────

def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def filter_frequency_range(notes: list[MidiNote],
                           fmin: float = 80.0,
                           fmax: float = 1000.0) -> list[MidiNote]:
    """Keep only notes within frequency range."""
    return [n for n in notes if fmin <= midi_to_hz(n.note) <= fmax]


def filter_melody_only(notes: list[MidiNote],
                       resolution_sec: float = 0.01) -> list[MidiNote]:
    """Keep only the highest-pitched note at each timestep.

    Quantizes time into bins of `resolution_sec` and picks the highest
    note active in each bin. Reconstructs note events from the surviving bins.
    """
    if not notes:
        return []

    max_time = max(n.end_sec for n in notes)
    n_bins = int(max_time / resolution_sec) + 1

    # For each time bin, find the highest active note
    best_note = [(-1, -1)] * n_bins  # (midi_note, original_index)

    for idx, n in enumerate(notes):
        start_bin = int(n.start_sec / resolution_sec)
        end_bin = int(n.end_sec / resolution_sec)
        for b in range(start_bin, min(end_bin, n_bins)):
            if n.note > best_note[b][0]:
                best_note[b] = (n.note, idx)

    # Reconstruct note events from surviving bins
    melody: list[MidiNote] = []
    current_note = -1
    current_start = 0.0
    current_idx = -1

    for b in range(n_bins):
        note_num, orig_idx = best_note[b]
        if note_num != current_note:
            if current_note >= 0:
                melody.append(MidiNote(
                    note=current_note,
                    start_sec=current_start,
                    end_sec=b * resolution_sec,
                    velocity=notes[current_idx].velocity,
                ))
            current_note = note_num
            current_start = b * resolution_sec
            current_idx = orig_idx

    if current_note >= 0:
        melody.append(MidiNote(
            note=current_note,
            start_sec=current_start,
            end_sec=n_bins * resolution_sec,
            velocity=notes[current_idx].velocity,
        ))

    # Remove very short fragments
    melody = [n for n in melody if n.end_sec - n.start_sec >= 0.05]
    return melody


def apply_offset(notes: list[MidiNote], offset_sec: float) -> list[MidiNote]:
    """Shift all note times by offset. Clips to t >= 0."""
    shifted = []
    for n in notes:
        s = n.start_sec - offset_sec
        e = n.end_sec - offset_sec
        if e > 0:
            shifted.append(MidiNote(
                note=n.note,
                start_sec=max(0.0, s),
                end_sec=e,
                velocity=n.velocity,
            ))
    return shifted


def trim_to_overlap(gt: list[MidiNote],
                    test: list[MidiNote]) -> tuple[list[MidiNote], list[MidiNote]]:
    """Trim both note lists to only the overlapping time range."""
    if not gt or not test:
        return gt, test

    gt_start = min(n.start_sec for n in gt)
    gt_end = max(n.end_sec for n in gt)
    test_start = min(n.start_sec for n in test)
    test_end = max(n.end_sec for n in test)

    overlap_start = max(gt_start, test_start)
    overlap_end = min(gt_end, test_end)

    if overlap_start >= overlap_end:
        print("WARNING: No overlapping time range between ground truth and test MIDI!")
        return gt, test

    def clip(notes: list[MidiNote]) -> list[MidiNote]:
        clipped = []
        for n in notes:
            if n.end_sec > overlap_start and n.start_sec < overlap_end:
                clipped.append(MidiNote(
                    note=n.note,
                    start_sec=max(n.start_sec, overlap_start),
                    end_sec=min(n.end_sec, overlap_end),
                    velocity=n.velocity,
                ))
        return clipped

    return clip(gt), clip(test)


# ─── Evaluation metrics ───────────────────────────────────────────────────────

@dataclass
class EvalResult:
    name: str
    n_detected: int
    n_ground_truth: int
    true_positives: int
    precision: float
    recall: float
    f1: float


def evaluate_notes(gt: list[MidiNote], detected: list[MidiNote],
                   pitch_tol: int = 1, onset_tol: float = 0.1,
                   octave_tolerant: bool = False) -> tuple[int, int, int]:
    """Match detected notes against ground truth.

    A detected note is a true positive if:
      - pitch is within ±pitch_tol semitones of a GT note
        (or same pitch class if octave_tolerant=True)
      - onset is within ±onset_tol seconds of a GT note
      - that GT note hasn't been matched yet (greedy, closest onset first)

    Returns (true_positives, n_detected, n_ground_truth).
    """
    if not gt or not detected:
        return 0, len(detected), len(gt)

    gt_matched = [False] * len(gt)
    tp = 0

    # Sort detected by onset for consistent matching
    det_sorted = sorted(enumerate(detected), key=lambda x: x[1].start_sec)

    for _det_idx, d in det_sorted:
        best_dist = float("inf")
        best_gt_idx = -1

        for gi, g in enumerate(gt):
            if gt_matched[gi]:
                continue
            # Pitch matching
            if octave_tolerant:
                # Same pitch class (note name), any octave, within ±pitch_tol
                pc_diff = abs((d.note % 12) - (g.note % 12))
                pc_diff = min(pc_diff, 12 - pc_diff)  # handle wrap-around
                if pc_diff > pitch_tol:
                    continue
            else:
                if abs(d.note - g.note) > pitch_tol:
                    continue
            onset_diff = abs(d.start_sec - g.start_sec)
            if onset_diff > onset_tol:
                continue
            if onset_diff < best_dist:
                best_dist = onset_diff
                best_gt_idx = gi

        if best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            tp += 1

    return tp, len(detected), len(gt)


def compute_metrics(name: str, gt: list[MidiNote], detected: list[MidiNote],
                    pitch_tol: int = 1, onset_tol: float = 0.1,
                    octave_tolerant: bool = False) -> EvalResult:
    """Compute precision, recall, F1 for a set of detected notes vs ground truth."""
    tp, n_det, n_gt = evaluate_notes(gt, detected, pitch_tol, onset_tol, octave_tolerant)

    precision = tp / n_det if n_det > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvalResult(
        name=name,
        n_detected=n_det,
        n_ground_truth=n_gt,
        true_positives=tp,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ─── Display ──────────────────────────────────────────────────────────────────

def print_results(results: list[EvalResult], pitch_tol: int, onset_tol: float):
    """Print a formatted comparison table."""
    print()
    print("=" * 72)
    print("  MIDI Extraction Accuracy Report")
    print(f"  Matching tolerance: ±{pitch_tol} semitone(s), ±{onset_tol * 1000:.0f} ms onset")
    print("=" * 72)
    print()

    # Header
    print(f"  {'System':<22} {'Notes':>6} {'TP':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-' * 22} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8}")

    for r in results:
        print(f"  {r.name:<22} {r.n_detected:>6} {r.true_positives:>6} "
              f"{r.precision:>7.1%} {r.recall:>7.1%} {r.f1:>7.1%}")

    print()
    print(f"  Ground truth notes: {results[0].n_ground_truth}")
    print()

    # Interpretation
    if len(results) == 2:
        ours, bp = results
        if ours.f1 > bp.f1:
            diff = ours.f1 - bp.f1
            print(f"  >> Our pipeline beats Basic Pitch by {diff:.1%} F1 on this input.")
        elif bp.f1 > ours.f1:
            diff = bp.f1 - ours.f1
            print(f"  >> Basic Pitch leads by {diff:.1%} F1 on this input.")
        else:
            print(f"  >> Tied on F1.")

        if ours.precision > bp.precision:
            print(f"  >> Our pipeline has higher precision "
                  f"({ours.precision:.1%} vs {bp.precision:.1%}) — fewer false notes.")
        if ours.recall > bp.recall:
            print(f"  >> Our pipeline has higher recall "
                  f"({ours.recall:.1%} vs {bp.recall:.1%}) — finds more correct notes.")
        print()

    print("=" * 72)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Evaluate MIDI extraction accuracy against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # MIDI files
    p.add_argument("--gt", required=True,
                   help="Ground truth MIDI (Basic Pitch on CLEAN audio)")
    p.add_argument("--ours", required=True,
                   help="Our pipeline's MIDI output (on NOISY audio)")
    p.add_argument("--bp",
                   help="Basic Pitch MIDI (on NOISY audio) — optional, for comparison")

    # Alignment
    p.add_argument("--offset", type=float, default=None,
                   help="Manual time offset in seconds (GT is shifted by -offset)")
    p.add_argument("--clean-audio",
                   help="Clean audio file for auto-alignment via cross-correlation")
    p.add_argument("--noisy-audio",
                   help="Noisy audio file for auto-alignment via cross-correlation")

    # Filtering
    p.add_argument("--melody-only", action="store_true",
                   help="Filter ground truth to melody (highest note per timestep)")
    p.add_argument("--freq-min", type=float, default=80.0,
                   help="Min frequency in Hz for ground truth filtering (default: 80)")
    p.add_argument("--freq-max", type=float, default=1000.0,
                   help="Max frequency in Hz for ground truth filtering (default: 1000)")

    # Tolerances
    p.add_argument("--pitch-tol", type=int, default=1,
                   help="Pitch tolerance in semitones (default: 1)")
    p.add_argument("--onset-tol", type=float, default=0.1,
                   help="Onset tolerance in seconds (default: 0.1)")
    p.add_argument("--octave-tolerant", action="store_true",
                   help="Match by pitch class (note name) regardless of octave")
    p.add_argument("--sweep-offset", action="store_true",
                   help="Auto-find best time offset by sweeping -5s to +5s")

    # Output
    p.add_argument("--csv", default=None,
                   help="Export results to CSV file (appends if file exists)")

    args = p.parse_args()

    # --- Load MIDIs ---
    print(f"Loading ground truth: {args.gt}")
    gt_notes = parse_midi(args.gt)
    print(f"  → {len(gt_notes)} notes")

    print(f"Loading our output:   {args.ours}")
    our_notes = parse_midi(args.ours)
    print(f"  → {len(our_notes)} notes")

    bp_notes = None
    if args.bp:
        print(f"Loading Basic Pitch:  {args.bp}")
        bp_notes = parse_midi(args.bp)
        print(f"  → {len(bp_notes)} notes")

    # --- Alignment ---
    offset = 0.0

    if args.sweep_offset:
        # Auto-find best offset by sweeping
        print("\nSweeping offsets to find best alignment...")
        best_offset = 0.0
        best_tp = 0
        # Coarse sweep: -10s to +10s in 0.5s steps
        for test_offset in [x * 0.5 for x in range(-20, 21)]:
            shifted = apply_offset(gt_notes[:], test_offset)
            tp, _, _ = evaluate_notes(shifted, our_notes,
                                      args.pitch_tol, args.onset_tol,
                                      args.octave_tolerant)
            if tp > best_tp:
                best_tp = tp
                best_offset = test_offset
        # Fine sweep: ±1s around best in 0.1s steps
        for test_offset in [best_offset + x * 0.1 for x in range(-10, 11)]:
            shifted = apply_offset(gt_notes[:], test_offset)
            tp, _, _ = evaluate_notes(shifted, our_notes,
                                      args.pitch_tol, args.onset_tol,
                                      args.octave_tolerant)
            if tp > best_tp:
                best_tp = tp
                best_offset = test_offset
        offset = best_offset
        print(f"Best offset: {offset:+.3f}s ({best_tp} matches)")
    elif args.offset is not None:
        offset = args.offset
        print(f"\nManual offset: {offset:+.3f}s")
    elif args.clean_audio and args.noisy_audio:
        offset = compute_offset(args.clean_audio, args.noisy_audio)
    else:
        print("\nNo alignment (use --offset or --clean-audio + --noisy-audio if needed)")

    if offset != 0.0:
        gt_notes = apply_offset(gt_notes, offset)
        print(f"Ground truth shifted by {-offset:+.3f}s → {len(gt_notes)} notes remain")

    # --- Filter ground truth ---
    gt_notes = filter_frequency_range(gt_notes, args.freq_min, args.freq_max)
    print(f"After frequency filter ({args.freq_min}-{args.freq_max} Hz): {len(gt_notes)} GT notes")

    if args.melody_only:
        gt_notes = filter_melody_only(gt_notes)
        print(f"After melody-only filter: {len(gt_notes)} GT notes")

    # Also filter detected notes to same frequency range
    our_notes = filter_frequency_range(our_notes, args.freq_min, args.freq_max)
    if bp_notes is not None:
        bp_notes = filter_frequency_range(bp_notes, args.freq_min, args.freq_max)

    # --- Trim to overlapping region ---
    gt_for_ours, our_trimmed = trim_to_overlap(gt_notes, our_notes)

    results = []

    # Evaluate ours
    our_result = compute_metrics("Digital Ear (ours)", gt_for_ours, our_trimmed,
                                 args.pitch_tol, args.onset_tol, args.octave_tolerant)
    results.append(our_result)

    # Evaluate Basic Pitch if provided
    if bp_notes is not None:
        gt_for_bp, bp_trimmed = trim_to_overlap(gt_notes, bp_notes)
        bp_result = compute_metrics("Basic Pitch (noisy)", gt_for_bp, bp_trimmed,
                                     args.pitch_tol, args.onset_tol, args.octave_tolerant)
        results.append(bp_result)

    print_results(results, args.pitch_tol, args.onset_tol)

    # --- CSV export ---
    if args.csv:
        import csv
        import os
        file_exists = os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Input", "System", "Notes Detected", "Ground Truth Notes",
                    "True Positives", "Precision", "Recall", "F1",
                    "Pitch Tolerance (st)", "Onset Tolerance (ms)",
                    "Melody Only", "Freq Range (Hz)"
                ])
            # Figure out input name from file paths
            input_name = os.path.basename(args.ours).replace(".mid", "")
            for r in results:
                writer.writerow([
                    input_name,
                    r.name,
                    r.n_detected,
                    r.n_ground_truth,
                    r.true_positives,
                    f"{r.precision:.4f}",
                    f"{r.recall:.4f}",
                    f"{r.f1:.4f}",
                    args.pitch_tol,
                    int(args.onset_tol * 1000),
                    args.melody_only,
                    f"{args.freq_min}-{args.freq_max}",
                ])
        print(f"\nResults appended to: {args.csv}")


if __name__ == "__main__":
    main()
