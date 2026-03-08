#!/usr/bin/env python3
"""
The Digital Ear -- streaming audio-to-MIDI melody extraction.

Reads audio in <=2048-sample blocks, runs each through
HPSS -> pitch detection -> online Viterbi -> note tracking,
then drops the block. ~31 MB constant memory.
"""
from __future__ import annotations

from digital_ear.audio_io import probe_sample_rate, stream_m4a_blocks_ffmpeg
from digital_ear.perf import PerfLogger
from digital_ear.preprocess import Preprocessor, rms
from digital_ear.features import FrameExtractor
from digital_ear.harmonic_pitch import HarmonicPitchDetector
from digital_ear.note_tracker import NoteTracker, NoteEvent
from digital_ear.melody_extractor import MelodyExtractor
from digital_ear.hpss import HPSS
from digital_ear.midi_writer import write_midi, MIDINote, seconds_to_ticks

import argparse, math, os, struct, sys, time, wave
from collections import deque
from dataclasses import dataclass
import numpy as np


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _midi_to_name(midi_num: int) -> str:
    """MIDI note number -> name, e.g. 60 -> C4."""
    octave = (midi_num // 12) - 1
    name = _NOTE_NAMES[midi_num % 12]
    return f"{name}{octave}"



@dataclass(frozen=True)
class Args:
    in_path: str
    out_path: str
    block_size: int
    target_sr: int
    dc_fc: float
    hp_fc: float
    lp_fc: float
    n_fft: int
    hop: int
    fmin: float
    fmax: float
    conf_th: float
    rms_th: float
    dump_frames: str
    debug: bool
    poly: bool
    dual: str
    wav: str
    min_note_sec: float
    melody_prog: int
    bg_prog: int


def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(
        prog="the-digital-ear",
        description="Paradromics Digital Ear, streaming audio to MIDI.",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input audio file (.m4a)")
    p.add_argument("--out", dest="out_path", required=True, help="Output MIDI file (.mid)")
    p.add_argument("--block", dest="block_size", type=int, default=2048, help="Max block size (samples)")
    p.add_argument("--sr", dest="target_sr", type=int, default=44100, help="Target decode sample rate in Hz.")
    p.add_argument("--dc", dest="dc_fc", type=float, default=30.0, help="DC blocker cutoff Hz")
    p.add_argument("--hp", dest="hp_fc", type=float, default=60.0, help="High-pass cutoff Hz")
    p.add_argument("--lp", dest="lp_fc", type=float, default=4000.0, help="Low-pass cutoff Hz")
    p.add_argument("--nfft", dest="n_fft", type=int, default=2048, help="FFT size (analysis frame length)")
    p.add_argument("--hop", dest="hop", type=int, default=512, help="Hop size (samples between frames)")
    p.add_argument("--fmin", type=float, default=80.0, help="Min f0 search (Hz)")
    p.add_argument("--fmax", type=float, default=1000.0, help="Max f0 search (Hz)")
    p.add_argument("--conf-th", dest="conf_th", type=float, default=7.0, help="Confidence threshold")
    p.add_argument("--rms-th", type=float, default=0.003, help="RMS threshold for voiced frames")
    p.add_argument("--dump-frames", type=str, default="", help="Optional CSV path to dump per-frame (debug)")
    p.add_argument("--debug", action="store_true", help="Print extra debug info")
    p.add_argument("--poly", action="store_true", help="Polyphonic MIDI output (multiple simultaneous notes)")
    p.add_argument("--dual", type=str, default="", help="Output stereo WAV: left=original, right=synthesized MIDI")
    p.add_argument("--wav", type=str, default="", help="Export MIDI as standalone synthesized WAV file")
    p.add_argument("--min-note-sec", dest="min_note_sec", type=float, default=0.15, help="Minimum note duration in seconds")
    p.add_argument("--melody-prog", dest="melody_prog", type=int, default=0, help="GM program for melody channel (0=piano)")
    p.add_argument("--bg-prog", dest="bg_prog", type=int, default=26, help="GM program for background poly channels (26=steel guitar)")

    ns = p.parse_args(argv)

    if ns.block_size <= 0:
        raise SystemExit("ERROR: --block must be positive.")
    if ns.block_size > 2048:
        raise SystemExit("ERROR: --block must be <= 2048.")
    if not os.path.exists(ns.in_path):
        raise SystemExit(f"ERROR: input file not found: {ns.in_path}")
    if ns.target_sr <= 0:
        raise SystemExit("ERROR: --sr must be positive.")
    if not ns.out_path.lower().endswith(".mid"):
        raise SystemExit("ERROR: --out must end with .mid")
    if ns.n_fft <= 0:
        raise SystemExit("ERROR: --nfft must be positive.")
    if ns.hop <= 0:
        raise SystemExit("ERROR: --hop must be positive.")
    if ns.hop > ns.n_fft:
        raise SystemExit("ERROR: --hop must be <= --nfft.")
    if ns.n_fft > 2048:
        raise SystemExit("ERROR: --nfft must be <= 2048 for this qualifier MVP.")
    if ns.fmin <= 0 or ns.fmax <= 0 or ns.fmax <= ns.fmin:
        raise SystemExit("ERROR: require 0 < --fmin < --fmax.")
    if ns.conf_th <= 0:
        raise SystemExit("ERROR: --conf-th must be positive.")
    if ns.rms_th < 0:
        raise SystemExit("ERROR: --rms-th must be >= 0.")
    if ns.min_note_sec <= 0:
        raise SystemExit("ERROR: --min-note-sec must be positive.")

    return Args(
        in_path=ns.in_path,
        out_path=ns.out_path,
        block_size=ns.block_size,
        target_sr=ns.target_sr,
        dc_fc=ns.dc_fc,
        hp_fc=ns.hp_fc,
        lp_fc=ns.lp_fc,
        n_fft=ns.n_fft,
        hop=ns.hop,
        fmin=ns.fmin,
        fmax=ns.fmax,
        conf_th=ns.conf_th,
        rms_th=ns.rms_th,
        dump_frames=ns.dump_frames,
        debug=ns.debug,
        poly=ns.poly,
        dual=ns.dual,
        wav=ns.wav,
        min_note_sec=ns.min_note_sec,
        melody_prog=ns.melody_prog,
        bg_prog=ns.bg_prog,
    )


def _feed_poly_voices(
    primary_hz: float | None,
    cand_buf: deque,
    poly_melodies: list[MelodyExtractor],
    poly_trackers: list[NoteTracker],
    poly_note_buf: list[list[NoteEvent]],
) -> None:
    """Pop oldest buffered candidates and feed filtered results to secondary voices."""
    if not cand_buf:
        return
    old_cands, old_ton = cand_buf.popleft()

    # exclude primary + already-assigned secondary pitches
    excluded: list[float | None] = [primary_hz]

    for vi, (sec_mel, sec_trk) in enumerate(zip(poly_melodies, poly_trackers)):
        # skip pitches within 2 semitones of excluded voices
        filtered: list[tuple[float, float]] = []
        for hz, conf in old_cands:
            too_close = False
            for ex_hz in excluded:
                if ex_hz is not None and ex_hz > 0:
                    try:
                        if abs(12.0 * math.log2(hz / ex_hz)) <= 2.0:
                            too_close = True
                            break
                    except (ValueError, ZeroDivisionError):
                        pass
            if not too_close:
                filtered.append((hz, conf))

        sec_f0 = sec_mel.push(filtered, old_ton)
        if sec_f0 is not None:
            poly_note_buf[vi].extend(sec_trk.push(sec_f0))
            excluded.append(sec_f0)


def _write_synth_wav(
    wav_path: str,
    note_events: list[NoteEvent],
    sr: int,
) -> None:
    """Synthesize note events to a mono WAV."""
    if not note_events:
        return
    end_sec = max(e.end_sec for e in note_events)
    n_samples = int(end_sec * sr) + sr  # +1s for release tail
    synth = np.zeros(n_samples, dtype=np.float32)
    attack = int(0.01 * sr)
    release = int(0.03 * sr)

    for evt in note_events:
        freq = 440.0 * (2.0 ** ((evt.note - 69) / 12.0))
        start = max(0, min(int(evt.start_sec * sr), n_samples))
        end = max(start, min(int(evt.end_sec * sr), n_samples))
        length = end - start
        if length <= 0:
            continue
        t = np.arange(length, dtype=np.float32) / sr
        tone = 0.4 * np.sin(2.0 * np.pi * freq * t)
        env = np.ones(length, dtype=np.float32)
        att = min(attack, length)
        rel = min(release, length - att)
        if att > 0:
            env[:att] = np.linspace(0.0, 1.0, att)
        if rel > 0:
            env[-rel:] = np.linspace(1.0, 0.0, rel)
        synth[start:end] += tone * env

    # normalize to -3dBFS
    peak = np.abs(synth).max()
    if peak > 0:
        synth *= 0.7 / peak

    pcm = np.clip(synth * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _write_dual_wav(
    dual_path: str,
    in_path: str,
    note_events: list[NoteEvent],
    sr: int,
    block_size: int,
) -> None:
    """Stereo WAV: left=original, right=synth."""
    _, blocks = stream_m4a_blocks_ffmpeg(in_path, block_size, target_sr=sr)
    chunks: list[np.ndarray] = []
    for blk in blocks:
        chunks.append(blk.copy())
    original = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    n_samples = len(original)
    duration = n_samples / float(sr)

    # synth right channel
    synth = np.zeros(n_samples, dtype=np.float32)
    attack = int(0.01 * sr)
    release = int(0.03 * sr)

    for evt in note_events:
        freq = 440.0 * (2.0 ** ((evt.note - 69) / 12.0))
        start = int(evt.start_sec * sr)
        end = int(evt.end_sec * sr)
        start = max(0, min(start, n_samples))
        end = max(start, min(end, n_samples))
        length = end - start
        if length <= 0:
            continue

        t = np.arange(length, dtype=np.float32) / sr
        tone = 0.4 * np.sin(2.0 * np.pi * freq * t)

        # envelope
        env = np.ones(length, dtype=np.float32)
        att = min(attack, length)
        rel = min(release, length - att)
        if att > 0:
            env[:att] = np.linspace(0.0, 1.0, att)
        if rel > 0:
            env[-rel:] = np.linspace(1.0, 0.0, rel)

        synth[start:end] += tone * env

    # normalize to -3dBFS
    orig_peak = np.abs(original).max()
    if orig_peak > 0:
        original *= 0.7 / orig_peak

    synth_peak = np.abs(synth).max()
    if synth_peak > 0:
        synth *= 0.7 / synth_peak

    # write stereo wav
    with wave.open(dual_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)

        # interleave L/R
        stereo = np.empty(n_samples * 2, dtype=np.float32)
        stereo[0::2] = original
        stereo[1::2] = synth

        # to int16
        pcm = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    perf = PerfLogger()
    perf.start()

    _stage_times: dict[str, float] = {}
    _stage_clock = time.monotonic()

    def _mark_stage(name: str) -> None:
        nonlocal _stage_clock
        now = time.monotonic()
        _stage_times[name] = now - _stage_clock
        _stage_clock = now

    print("STAGE=decoding", flush=True)
    input_sr = probe_sample_rate(args.in_path)
    out_sr, blocks = stream_m4a_blocks_ffmpeg(args.in_path, args.block_size, target_sr=args.target_sr)

    # pipeline setup
    pre = Preprocessor(fs=float(out_sr), dc_fc=args.dc_fc, hp_fc=args.hp_fc, lp_fc=args.lp_fc)
    hpss = HPSS(n_fft=args.n_fft, hop=args.hop, kernel_h=31, kernel_p=31, power=2.0)
    fx = FrameExtractor(n_fft=args.n_fft, hop=args.hop)
    det = HarmonicPitchDetector(
        sr=float(out_sr),
        n_fft=args.n_fft,
        hop=args.hop,
        fmin=args.fmin,
        fmax=args.fmax,
        conf_threshold=args.conf_th,
    )

    hop_sec = args.hop / float(out_sr)
    n_top = 5 if not args.poly else 8

    # HPSS drops first kernel_h//2 frames, so we correct for that offset
    # plus the Hann window centre shift. Negated because NoteTracker subtracts it.
    hpss_delay_sec = (hpss.kernel_h // 2) * hop_sec
    latency_sec = args.n_fft / (2.0 * float(out_sr)) - hpss_delay_sec
    melody = MelodyExtractor(hop_sec=hop_sec)
    tracker = NoteTracker(
        hop_sec=hop_sec, median_window=15,
        min_note_sec=args.min_note_sec, merge_gap_sec=0.12, latency_sec=latency_sec,
    )

    # poly mode -- secondary voices
    n_poly_voices = 2 if args.poly else 0
    poly_melodies: list[MelodyExtractor] = []
    poly_trackers: list[NoteTracker] = []
    for _ in range(n_poly_voices):
        poly_melodies.append(MelodyExtractor(hop_sec=hop_sec))
        poly_trackers.append(NoteTracker(
            hop_sec=hop_sec, median_window=15,
            min_note_sec=0.20, merge_gap_sec=0.15, latency_sec=latency_sec,
        ))
    # buffer candidates during primary Viterbi lag for secondary voice filtering
    cand_buf: deque[tuple[list[tuple[float, float]], float]] = deque()
    # poly note accumulator
    poly_note_bufs: list[list[NoteEvent]] = [[] for _ in range(n_poly_voices)]

    csv_f = None
    if args.dump_frames:
        csv_f = open(args.dump_frames, "w", encoding="utf-8")
        csv_f.write("frame_idx,time_sec,f0_hz,conf,rms,voiced\n")

    # buffer for CSV dump
    csv_buf: deque[tuple[int, float, list[tuple[float,float]], float]] = deque()

    block_count = 0
    total_samples = 0
    max_len_seen = 0
    frame_count = 0
    stream_sample_index = 0
    melody_frame_count = 0

    note_events: list[NoteEvent] = []

    # --- streaming loop ---
    _mark_stage("decoding")
    print("STAGE=processing", flush=True)

    # debug stats
    _voiced_frames = 0
    _unvoiced_frames = 0
    _conf_sum = 0.0
    for blk in blocks:
        perf.sample()
        block_count += 1
        n = int(blk.shape[0])
        total_samples += n
        if n > max_len_seen:
            max_len_seen = n
        assert n <= args.block_size, f"Block too large: {n} > {args.block_size}"

        # preprocess (dc block + hpf + lpf)
        y = pre.process(blk)

        # hpss
        hpss.push(y)

        # consume harmonic frames
        for harmonic_block in hpss.pop_harmonic():
            # extract frames
            for frame, frame_start in fx.push_indexed(harmonic_block, stream_sample_index):
                frame_rms = rms(frame)

                if frame_rms < args.rms_th:
                    candidates = []
                    tonality = 0.0
                else:
                    candidates, tonality = det.estimate_candidates(frame, n_top=n_top, min_conf=1.5)

                # viterbi melody (primary sees top-5 regardless of poly mode)
                f0_hz = melody.push(candidates[:5], tonality)

                # csv dump
                if csv_f is not None:
                    csv_buf.append((frame_count, frame_start / float(out_sr), candidates, frame_rms))

                # poly candidate buffer
                if args.poly:
                    cand_buf.append((candidates, tonality))

                # note tracking (delayed by viterbi lag)
                if f0_hz is not None:
                    note_events.extend(tracker.push(f0_hz))
                    melody_frame_count += 1

                    # write CSV row for oldest buffered frame
                    if csv_f is not None and csv_buf:
                        fi, t_sec, cands, frms = csv_buf.popleft()
                        top_hz = f0_hz if f0_hz else 0.0
                        top_conf = 0.0
                        if cands:
                            top_conf = cands[0][1]
                        voiced = 1 if f0_hz and f0_hz > 0 else 0
                        csv_f.write(f"{fi},{t_sec:.4f},{top_hz:.2f},{top_conf:.2f},{frms:.6f},{voiced}\n")

                    # feed secondary voices
                    if args.poly and cand_buf:
                        _feed_poly_voices(
                            f0_hz, cand_buf, poly_melodies, poly_trackers,
                            poly_note_bufs,
                        )

                # voicing stats
                if candidates:
                    _voiced_frames += 1
                    _conf_sum += candidates[0][1]
                else:
                    _unvoiced_frames += 1

                frame_count += 1

            stream_sample_index += len(harmonic_block)

    # ---- flush pipeline ----
    _mark_stage("processing")
    print("STAGE=flushing", flush=True)
    perf.sample()

    # flush hpss
    for harmonic_block in hpss.flush():
        for frame, frame_start in fx.push_indexed(harmonic_block, stream_sample_index):
            frame_rms = rms(frame)
            if frame_rms < args.rms_th:
                candidates, tonality = [], 0.0
            else:
                candidates, tonality = det.estimate_candidates(frame, n_top=n_top, min_conf=1.5)

            f0_hz = melody.push(candidates[:5], tonality)
            if args.poly:
                cand_buf.append((candidates, tonality))
            if f0_hz is not None:
                note_events.extend(tracker.push(f0_hz))
                melody_frame_count += 1
                if args.poly and cand_buf:
                    _feed_poly_voices(f0_hz, cand_buf, poly_melodies, poly_trackers, poly_note_bufs)
            frame_count += 1
        stream_sample_index += len(harmonic_block)

    # flush viterbi
    for f0_hz in melody.flush():
        note_events.extend(tracker.push(f0_hz))
        melody_frame_count += 1
        if csv_f is not None and csv_buf:
            fi, t_sec, cands, frms = csv_buf.popleft()
            top_hz = f0_hz if f0_hz else 0.0
            top_conf = cands[0][1] if cands else 0.0
            voiced = 1 if f0_hz and f0_hz > 0 else 0
            csv_f.write(f"{fi},{t_sec:.4f},{top_hz:.2f},{top_conf:.2f},{frms:.6f},{voiced}\n")
        if args.poly and cand_buf:
            _feed_poly_voices(f0_hz, cand_buf, poly_melodies, poly_trackers, poly_note_bufs)

    # flush secondary voices
    poly_events: list[NoteEvent] = []
    for vi in range(n_poly_voices):
        # notes collected so far
        vi_notes: list[NoteEvent] = poly_note_bufs[vi]
        # drain leftover candidates
        while cand_buf:
            old_cands, old_ton = cand_buf.popleft()
            sec_f0 = poly_melodies[vi].push(old_cands, old_ton)
            if sec_f0 is not None:
                vi_notes.extend(poly_trackers[vi].push(sec_f0))
        for sec_f0 in poly_melodies[vi].flush():
            vi_notes.extend(poly_trackers[vi].push(sec_f0))
        vi_notes.extend(poly_trackers[vi].flush())
        for evt in vi_notes:
            evt._voice = vi  # tag with voice index for channel assignment
        poly_events.extend(vi_notes)

    # flush tracker
    note_events.extend(tracker.flush())

    perf.sample()

    # midi output
    _mark_stage("flushing")
    print("STAGE=writing_midi", flush=True)
    ticks_per_beat = 480
    bpm = 120.0
    midi_notes: list[MIDINote] = []
    for evt in note_events:
        midi_notes.append(MIDINote(
            note=evt.note,
            start_tick=seconds_to_ticks(evt.start_sec, ticks_per_beat, bpm),
            end_tick=seconds_to_ticks(evt.end_sec, ticks_per_beat, bpm),
            velocity=evt.velocity,
        ))

    # bg voices quieter
    for evt in poly_events:
        vi = getattr(evt, '_voice', 0)
        ch = vi + 1
        vel = 40 if vi == 0 else 30
        midi_notes.append(MIDINote(
            note=evt.note,
            start_tick=seconds_to_ticks(evt.start_sec, ticks_per_beat, bpm),
            end_tick=seconds_to_ticks(evt.end_sec, ticks_per_beat, bpm),
            velocity=vel,
            channel=ch,
        ))

    midi_notes.sort(key=lambda m: m.start_tick)
    programs = {0: args.melody_prog, 1: args.bg_prog, 2: args.bg_prog}
    write_midi(midi_notes, ticks_per_beat, args.out_path, programs=programs)

    _mark_stage("writing_midi")

    # synth wav output
    if args.wav:
        print("STAGE=dual_wav", flush=True)
        all_events = list(note_events)
        if args.poly:
            all_events.extend(poly_events)
        _write_synth_wav(args.wav, all_events, out_sr)
        print(f"WAV_OUTPUT={args.wav}")
        _mark_stage("synth_wav")

    # --- dual output (stereo: left=original, right=synth)
    if args.dual:
        print("STAGE=dual_wav", flush=True)
        all_events = list(note_events)
        if args.poly:
            all_events.extend(poly_events)
        _write_dual_wav(args.dual, args.in_path, all_events, out_sr, args.block_size)
        print(f"DUAL_OUTPUT={args.dual}")
        _mark_stage("dual_wav")

    elapsed_s, peak_mb = perf.stop_and_report()

    print("STAGE=complete", flush=True)

    # reporting
    print(f"INPUT_FILE={args.in_path}")
    print(f"OUTPUT_FILE={args.out_path}")
    print(f"BLOCK_SIZE={args.block_size}")

    if input_sr is not None:
        print(f"INPUT_SAMPLE_RATE={input_sr}")
    else:
        print("INPUT_SAMPLE_RATE=UNKNOWN")

    print(f"TARGET_SAMPLE_RATE={args.target_sr}")
    print(f"DECODE_SAMPLE_RATE={out_sr}")
    print(f"BLOCKS_PROCESSED={block_count}")
    print(f"MAX_BLOCK_LEN={max_len_seen}")
    print(f"TOTAL_SAMPLES={total_samples}")
    print(f"FRAMES_PROCESSED={frame_count}")
    n_primary = len(note_events)
    n_poly = len(poly_events)
    print(f"MELODY_PROG={args.melody_prog}")
    if args.poly:
        print(f"BG_PROG={args.bg_prog}")
    print(f"NOTES_DETECTED={len(midi_notes)}")
    if args.poly:
        print(f"PRIMARY_NOTES={n_primary}")
        print(f"SECONDARY_NOTES={n_poly}")
    print(f"ELAPSED_SEC={elapsed_s:.6f}")
    print(f"PEAK_RSS_MB={peak_mb:.2f}")

    if args.debug:
        # note summary
        all_note_events = list(note_events) + list(poly_events)
        if all_note_events:
            lo = min(e.note for e in all_note_events)
            hi = max(e.note for e in all_note_events)
            durs = [e.end_sec - e.start_sec for e in all_note_events]
            avg_dur = sum(durs) / len(durs)
            print(f"\n-- Note Summary --")
            print(f"  Range  {_midi_to_name(lo)} ({lo}) → {_midi_to_name(hi)} ({hi})")
            print(f"  Count  {len(all_note_events)}  |  Avg duration {avg_dur:.3f}s")

        # voicing stats
        total_f = _voiced_frames + _unvoiced_frames
        if total_f > 0:
            pct = 100.0 * _voiced_frames / total_f
            avg_conf = _conf_sum / _voiced_frames if _voiced_frames > 0 else 0
            print(f"\nPitch & Voicing")
            print(f"  Voiced frames    {_voiced_frames}/{total_f} ({pct:.1f}%)")
            print(f"  Avg confidence   {avg_conf:.2f}")
            voiced_sec = _voiced_frames * hop_sec
            print(f"  Voiced time      {voiced_sec:.2f}s")

        # perf
        audio_dur = total_samples / float(out_sr) if out_sr > 0 else 0
        if audio_dur > 0:
            rtf = elapsed_s / audio_dur
            speed = audio_dur / elapsed_s if elapsed_s > 0 else 0
            print(f"\n-- Speed & Memory --")
            print(f"  Audio duration   {audio_dur:.2f}s")
            print(f"  Processing time  {elapsed_s:.2f}s")
            print(f"  Real-time factor {rtf:.2f}x  ({speed:.1f}x faster than real-time)")
            print(f"  Peak memory      {peak_mb:.1f} MB")
            # pass/fail check
            speed_ok = speed >= 4.0
            mem_ok = peak_mb < 500
            if speed_ok and mem_ok:
                print(f"\033[32m  ✓ PASS  {speed:.1f}x speed (≥4x) · {peak_mb:.1f} MB RAM (<500 MB)\033[0m")
            else:
                fails = []
                if not speed_ok:
                    fails.append(f"{speed:.1f}x speed (<4x)")
                else:
                    fails.append(f"{speed:.1f}x speed")
                if not mem_ok:
                    fails.append(f"{peak_mb:.1f} MB RAM (≥500 MB)")
                else:
                    fails.append(f"{peak_mb:.1f} MB RAM")
                print(f"\033[31m  ✗ FAIL  {' · '.join(fails)}\033[0m")

        # timing breakdown
        print(f"\nPipeline Timing")
        for stage, secs in _stage_times.items():
            bar_len = int(min(secs / max(elapsed_s, 0.001) * 30, 30))
            bar = "█" * bar_len
            print(f"  {stage:<14s} {secs:6.3f}s  {bar}")

    if csv_f is not None:
        csv_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
