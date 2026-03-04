#!/usr/bin/env python3
"""
Generate 4 individual spectrogram PNGs showing the pipeline progression:
  1) Raw audio (what old HPS saw)
  2) After HPSS harmonic separation (what MELODIA sees)
  3) HPSS spectrogram + extracted pitch contour overlay
  4) Synthesized MIDI spectrogram (the final output)

Outputs bare spectrograms — no titles, no axis labels, no colorbars.

Usage:
    python generate_spectrogram.py --in "Input 4.m4a"
    python generate_spectrogram.py --in "Input 4.m4a" --duration 15
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from digital_ear.audio_io import stream_m4a_blocks_ffmpeg
from digital_ear.preprocess import Preprocessor
from digital_ear.hpss import HPSS
from digital_ear.features import FrameExtractor
from digital_ear.harmonic_pitch import HarmonicPitchDetector
from digital_ear.melody_extractor import MelodyExtractor
from digital_ear.note_tracker import NoteTracker


def _rms(x):
    return float(np.sqrt(np.mean(x * x)))


def _save_spectrogram(spec_db, time_axis, freq_axis, vmin, vmax,
                      out_path, freq_lim, f0_contour=None):
    """Save a spectrogram PNG with minimal axes and colorbar legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Clip the top 8 dB to white — peaks saturate bright while
    # the full low-end range stays visible (no PowerNorm needed).
    im = ax.pcolormesh(time_axis, freq_axis, spec_db, shading="auto",
                       vmin=vmin, vmax=vmax - 8,
                       cmap="hot")

    if f0_contour is not None:
        voiced = [(t, f) for t, f in f0_contour if f is not None and f > 0]
        if voiced:
            times, freqs = zip(*voiced)
            ax.scatter(times, freqs, s=14, c="#00ffff", alpha=0.9,
                       linewidths=0, zorder=5, label="Extracted pitch")

    ax.set_yscale("log")
    ax.set_ylim(60, freq_lim)
    ax.set_xlim(time_axis[0], time_axis[-1])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("dB", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(labelcolor="white", labelsize=9)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.5)

    # Legend for pitch overlay
    if f0_contour is not None:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.6,
                  facecolor="black", edgecolor="white", labelcolor="white")

    # Clean axes with tick labels
    yticks = [100, 200, 400, 800, 1600]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v}" for v in yticks], color="white", fontsize=10)
    ax.minorticks_off()
    ax.set_ylabel("Hz", color="white", fontsize=11)
    ax.set_xlabel("Time (s)", color="white", fontsize=11)
    ax.tick_params(axis="x", colors="white", labelsize=10)
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.5)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                facecolor="black")
    plt.close(fig)


def _save_piano_roll(note_events, duration, out_path, freq_lim):
    """Save a piano-roll style MIDI visualization — solid cyan bars at note frequencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_facecolor("black")

    for evt in note_events:
        freq = 440.0 * (2.0 ** ((evt.note - 69) / 12.0))
        if freq < 60 or freq > freq_lim:
            continue
        # Bar height: ±1 semitone in log space
        f_lo = freq * 2 ** (-0.5 / 12)
        f_hi = freq * 2 ** (0.5 / 12)
        ax.fill_between(
            [evt.start_sec, evt.end_sec], f_lo, f_hi,
            color="#00ffff", alpha=0.9, linewidth=0
        )

    # Add a legend entry
    import matplotlib.patches as mpatches
    legend_patch = mpatches.Patch(color="#00ffff", alpha=0.9, label="MIDI notes")
    ax.legend(handles=[legend_patch], loc="upper right", fontsize=9,
              framealpha=0.6, facecolor="black", edgecolor="white", labelcolor="white")

    ax.set_yscale("log")
    ax.set_ylim(60, freq_lim)
    ax.set_xlim(0, duration)

    yticks = [100, 200, 400, 800, 1600]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v}" for v in yticks], color="white", fontsize=10)
    ax.minorticks_off()
    ax.set_ylabel("Hz", color="white", fontsize=11)
    ax.set_xlabel("Time (s)", color="white", fontsize=11)
    ax.tick_params(axis="x", colors="white", labelsize=10)
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.5)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                facecolor="black")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Generate 4 pipeline spectrogram PNGs")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--outdir", default="outputs/spectrograms")
    p.add_argument("--suffix", default="", help="Filename suffix e.g. '_inp3'")
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--duration", type=float, default=0,
                   help="Max seconds to show (0 = full file)")
    args = p.parse_args()

    sr = args.sr
    n_fft = 2048
    hop = 512
    hop_sec = hop / sr
    freq_limit_hz = 2000

    # --- Pipeline components ---
    pre = Preprocessor(fs=float(sr), dc_fc=30.0, hp_fc=60.0, lp_fc=4000.0)
    hpss = HPSS(n_fft=n_fft, hop=hop, kernel_h=31, kernel_p=31, power=2.0)
    fx = FrameExtractor(n_fft=n_fft, hop=hop)
    det = HarmonicPitchDetector(sr=float(sr), n_fft=n_fft, hop=hop,
                                fmin=80.0, fmax=1000.0, conf_threshold=7.0)
    melody = MelodyExtractor(hop_sec=hop_sec)
    tracker = NoteTracker(hop_sec=hop_sec, median_window=15,
                          min_note_sec=0.12, merge_gap_sec=0.12)

    window = np.hanning(n_fft).astype(np.float32)

    # Accumulators
    unproc_frames: list[np.ndarray] = []  # truly raw (no preprocessing)
    raw_frames: list[np.ndarray] = []      # after preprocessing

    # Unprocessed audio overlap buffer
    unproc_buf = np.zeros(n_fft, dtype=np.float32)
    unproc_fill = 0

    # Preprocessed audio overlap buffer
    raw_buf = np.zeros(n_fft, dtype=np.float32)
    raw_fill = 0

    # HPSS harmonic overlap buffer
    harm_frames: list[np.ndarray] = []
    harm_buf = np.zeros(n_fft, dtype=np.float32)
    harm_fill = 0

    # Pitch contour from Viterbi
    f0_contour: list[tuple[float, float | None]] = []
    frame_count = 0
    stream_sample_idx = 0

    max_frames = 0
    if args.duration > 0:
        max_frames = int(args.duration * sr / hop)

    out_sr, blocks = stream_m4a_blocks_ffmpeg(args.in_path, args.block, target_sr=sr)

    for blk in blocks:
        # --- Truly raw spectrogram (no preprocessing) ---
        u_idx = 0
        while u_idx < len(blk):
            space = n_fft - unproc_fill
            take = min(space, len(blk) - u_idx)
            unproc_buf[unproc_fill:unproc_fill + take] = blk[u_idx:u_idx + take]
            unproc_fill += take
            u_idx += take

            if unproc_fill == n_fft:
                mag = np.abs(np.fft.rfft(unproc_buf * window)).astype(np.float32)
                unproc_frames.append(mag)
                remain = n_fft - hop
                unproc_buf[:remain] = unproc_buf[hop:]
                unproc_buf[remain:] = 0.0
                unproc_fill = remain

        y = pre.process(blk)

        # --- Preprocessed spectrogram frames ---
        idx = 0
        while idx < len(y):
            space = n_fft - raw_fill
            take = min(space, len(y) - idx)
            raw_buf[raw_fill:raw_fill + take] = y[idx:idx + take]
            raw_fill += take
            idx += take

            if raw_fill == n_fft:
                mag = np.abs(np.fft.rfft(raw_buf * window)).astype(np.float32)
                raw_frames.append(mag)
                remain = n_fft - hop
                raw_buf[:remain] = raw_buf[hop:]
                raw_buf[remain:] = 0.0
                raw_fill = remain
                if max_frames and len(raw_frames) >= max_frames:
                    break

        # --- HPSS + pitch detection ---
        hpss.push(y)
        for harm_block in hpss.pop_harmonic():
            # Harmonic spectrogram frames
            h_idx = 0
            while h_idx < len(harm_block):
                space = n_fft - harm_fill
                take = min(space, len(harm_block) - h_idx)
                harm_buf[harm_fill:harm_fill + take] = harm_block[h_idx:h_idx + take]
                harm_fill += take
                h_idx += take

                if harm_fill == n_fft:
                    mag = np.abs(np.fft.rfft(harm_buf * window)).astype(np.float32)
                    harm_frames.append(mag)
                    remain = n_fft - hop
                    harm_buf[:remain] = harm_buf[hop:]
                    harm_buf[remain:] = 0.0
                    harm_fill = remain

            # Pitch detection on harmonic audio
            for frame, frame_start in fx.push_indexed(harm_block, stream_sample_idx):
                frame_rms = _rms(frame)
                if frame_rms < 0.005:
                    candidates, tonality = [], 0.0
                else:
                    candidates, tonality = det.estimate_candidates(frame, n_top=5, min_conf=1.5)

                f0_hz = melody.push(candidates, tonality)
                if f0_hz is not None:
                    t = frame_count * hop_sec
                    f0_contour.append((t, f0_hz if (f0_hz is not None and f0_hz > 0) else None))
                    tracker.push(f0_hz)
                    frame_count += 1
                elif f0_hz is None:
                    # Viterbi hasn't emitted yet (filling lag buffer)
                    pass

            stream_sample_idx += len(harm_block)

        if max_frames and len(raw_frames) >= max_frames:
            break

    # Flush HPSS
    for harm_block in hpss.flush():
        h_idx = 0
        while h_idx < len(harm_block):
            space = n_fft - harm_fill
            take = min(space, len(harm_block) - h_idx)
            harm_buf[harm_fill:harm_fill + take] = harm_block[h_idx:h_idx + take]
            harm_fill += take
            h_idx += take

            if harm_fill == n_fft:
                mag = np.abs(np.fft.rfft(harm_buf * window)).astype(np.float32)
                harm_frames.append(mag)
                remain = n_fft - hop
                harm_buf[:remain] = harm_buf[hop:]
                harm_buf[remain:] = 0.0
                harm_fill = remain

        for frame, frame_start in fx.push_indexed(harm_block, stream_sample_idx):
            frame_rms = _rms(frame)
            if frame_rms < 0.005:
                candidates, tonality = [], 0.0
            else:
                candidates, tonality = det.estimate_candidates(frame, n_top=5, min_conf=1.5)

            f0_hz = melody.push(candidates, tonality)
            if f0_hz is not None:
                t = frame_count * hop_sec
                f0_contour.append((t, f0_hz if (f0_hz is not None and f0_hz > 0) else None))
                tracker.push(f0_hz)
                frame_count += 1

        stream_sample_idx += len(harm_block)

    # Flush Viterbi
    for f0_hz in melody.flush():
        t = frame_count * hop_sec
        f0_val = f0_hz if (f0_hz is not None and f0_hz > 0) else None
        f0_contour.append((t, f0_val))
        if f0_hz is not None:
            tracker.push(f0_hz)
        frame_count += 1

    # Flush note tracker
    note_events = tracker.flush()

    if not raw_frames:
        print("ERROR: No frames extracted")
        return

    # --- Build spectrogram matrices ---
    # Build spectrogram matrices
    unproc_spec = np.array(unproc_frames).T
    unproc_spec_db = 20.0 * np.log10(np.maximum(unproc_spec, 1e-10))

    raw_spec = np.array(raw_frames).T
    raw_spec_db = 20.0 * np.log10(np.maximum(raw_spec, 1e-10))

    n_frames = min(raw_spec.shape[1], len(harm_frames), unproc_spec.shape[1])
    harm_spec = np.array(harm_frames[:n_frames]).T
    harm_spec_db = 20.0 * np.log10(np.maximum(harm_spec, 1e-10))
    raw_spec_db = raw_spec_db[:, :n_frames]
    unproc_spec_db = unproc_spec_db[:, :n_frames]

    # Dynamic range
    vmax = max(unproc_spec_db.max(), raw_spec_db.max(), harm_spec_db.max())
    vmin = vmax - 60

    # Frequency limit
    bin_limit = int(freq_limit_hz * n_fft / sr) + 1
    unproc_spec_db = unproc_spec_db[:bin_limit, :]
    raw_spec_db = raw_spec_db[:bin_limit, :]
    harm_spec_db = harm_spec_db[:bin_limit, :]

    # Skip DC bin for log scale
    unproc_spec_db = unproc_spec_db[1:, :]
    raw_spec_db = raw_spec_db[1:, :]
    harm_spec_db = harm_spec_db[1:, :]

    time_axis = np.arange(n_frames) * hop / sr
    freq_axis = (np.arange(bin_limit) * sr / n_fft)[1:]

    # --- Save 5 individual PNGs ---
    sfx = args.suffix
    os.makedirs(args.outdir, exist_ok=True)

    # 1: Truly raw decoded audio (no preprocessing)
    p1 = os.path.join(args.outdir, f"spec_1_raw{sfx}.png")
    _save_spectrogram(unproc_spec_db, time_axis, freq_axis,
                      vmin, vmax, p1, freq_limit_hz)
    print(f"Saved: {p1}")

    # 2: After preprocessing (DC block + HPF + LPF)
    p2 = os.path.join(args.outdir, f"spec_2_preprocess{sfx}.png")
    _save_spectrogram(raw_spec_db, time_axis, freq_axis,
                      vmin, vmax, p2, freq_limit_hz)
    print(f"Saved: {p2}")

    # 3: After HPSS
    p3 = os.path.join(args.outdir, f"spec_3_hpss{sfx}.png")
    _save_spectrogram(harm_spec_db, time_axis, freq_axis,
                      vmin, vmax, p3, freq_limit_hz)
    print(f"Saved: {p3}")

    # 4: HPSS + pitch contour overlay
    p4 = os.path.join(args.outdir, f"spec_4_pitch{sfx}.png")
    _save_spectrogram(harm_spec_db, time_axis, freq_axis,
                      vmin, vmax, p4, freq_limit_hz,
                      f0_contour=f0_contour)
    print(f"Saved: {p4}")

    # 5: Piano roll (solid bars at note frequencies)
    if note_events:
        p5 = os.path.join(args.outdir, f"spec_5_midi{sfx}.png")
        _save_piano_roll(note_events, time_axis[-1], p5, freq_limit_hz)
        print(f"Saved: {p5}")
    else:
        print("WARN: No note events — skipping MIDI piano roll")

    print(f"\n  Frames: {n_frames}")
    print(f"  Pitch contour points: {len(f0_contour)}")
    print(f"  Note events: {len(note_events)}")
    print(f"  Duration: {time_axis[-1]:.1f}s")


if __name__ == "__main__":
    main()
