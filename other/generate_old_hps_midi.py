#!/usr/bin/env python3
"""
Generate MIDI output using the OLD pipeline (HPS pitch detection, no HPSS,
no Viterbi, simple hysteresis note tracking) for A/B comparison with the
current pipeline.

Usage:
    python generate_old_hps_midi.py --in "Input 4.m4a" --out outputs/output4_OLD_HPS.mid
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

# Reuse the shared components that survived into the new pipeline
from digital_ear.audio_io import stream_m4a_blocks_ffmpeg
from digital_ear.preprocess import Preprocessor, rms
from digital_ear.features import FrameExtractor, hann_window, rfft_mag

# Old components
from digital_ear.old.pitch import FFTPitchDetector
from digital_ear.old.voicing import VoicingGate

# MIDI writer (same in both pipelines)
from digital_ear.midi_writer import write_midi, MIDINote, seconds_to_ticks


def hz_to_midi(hz: float) -> int:
    return int(round(69.0 + 12.0 * math.log2(hz / 440.0)))


def main():
    p = argparse.ArgumentParser(description="Old HPS pipeline for A/B comparison")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--conf-th", type=float, default=7.0)
    p.add_argument("--rms-th", type=float, default=0.003)
    p.add_argument("--min-note-sec", type=float, default=0.15)
    args = p.parse_args()

    t0 = time.perf_counter()

    n_fft = 2048
    hop = 512
    hop_sec = hop / float(args.sr)

    # --- Pipeline components (OLD versions) ---
    pre = Preprocessor(fs=float(args.sr), dc_fc=30.0, hp_fc=60.0, lp_fc=4000.0)
    fx = FrameExtractor(n_fft=n_fft, hop=hop)
    det = FFTPitchDetector(sr=float(args.sr), n_fft=n_fft, fmin=80.0, fmax=1000.0)
    gate = VoicingGate(conf_threshold=args.conf_th, rms_threshold=args.rms_th)
    win = hann_window(n_fft)

    # --- Simple hysteresis note tracker (the original Iteration 6 plan) ---
    # "Require same note for N consecutive frames to start, M unvoiced to end"
    ONSET_FRAMES = 3    # need 3 consecutive same-note frames to start a note
    OFFSET_FRAMES = 3   # need 3 consecutive unvoiced frames to end a note

    current_note = None
    current_start = 0.0
    pending_note = None
    pending_count = 0
    unvoiced_count = 0
    frame_count = 0

    note_events: list[tuple[int, float, float]] = []  # (midi, start, end)

    # --- Streaming loop ---
    out_sr, blocks = stream_m4a_blocks_ffmpeg(args.in_path, args.block, target_sr=args.sr)

    for blk in blocks:
        y = pre.process(blk)

        # No HPSS — feed raw preprocessed audio directly to frame extractor
        for frame in fx.push(y):
            frame_rms = rms(frame)

            # HPS pitch detection (single best, no candidates)
            f0_hz, conf = det.estimate(frame)

            # Simple voicing gate
            f0_hz, voiced = gate.apply(f0_hz, conf, frame_rms)

            t_sec = frame_count * hop_sec

            if voiced and f0_hz is not None:
                midi = hz_to_midi(f0_hz)
                midi = max(0, min(127, midi))
                unvoiced_count = 0

                if current_note is None:
                    # No active note — start pending
                    if pending_note == midi:
                        pending_count += 1
                        if pending_count >= ONSET_FRAMES:
                            current_note = midi
                            current_start = t_sec - (ONSET_FRAMES - 1) * hop_sec
                            pending_note = None
                            pending_count = 0
                    else:
                        pending_note = midi
                        pending_count = 1
                elif midi != current_note:
                    # Different note — end current, start pending
                    end_sec = t_sec
                    if end_sec - current_start >= args.min_note_sec:
                        note_events.append((current_note, current_start, end_sec))
                    current_note = None
                    pending_note = midi
                    pending_count = 1
                # else: same note, keep going
            else:
                unvoiced_count += 1
                pending_note = None
                pending_count = 0

                if current_note is not None and unvoiced_count >= OFFSET_FRAMES:
                    end_sec = t_sec - (OFFSET_FRAMES - 1) * hop_sec
                    if end_sec - current_start >= args.min_note_sec:
                        note_events.append((current_note, current_start, end_sec))
                    current_note = None

            frame_count += 1

    # Flush final note
    if current_note is not None:
        end_sec = frame_count * hop_sec
        if end_sec - current_start >= args.min_note_sec:
            note_events.append((current_note, current_start, end_sec))

    # --- Write MIDI ---
    ticks_per_beat = 480
    bpm = 120.0
    midi_notes = []
    for midi, start, end in note_events:
        midi_notes.append(MIDINote(
            note=midi,
            start_tick=seconds_to_ticks(start, ticks_per_beat, bpm),
            end_tick=seconds_to_ticks(end, ticks_per_beat, bpm),
            velocity=80,
        ))
    midi_notes.sort(key=lambda m: m.start_tick)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    write_midi(midi_notes, ticks_per_beat, args.out_path)

    elapsed = time.perf_counter() - t0
    print(f"OLD HPS Pipeline")
    print(f"  Notes detected: {len(note_events)}")
    print(f"  Frames: {frame_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: {args.out_path}")


if __name__ == "__main__":
    main()
