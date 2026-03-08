#!/usr/bin/env python3
"""
Compare Digital Ear MIDI outputs against Basic Pitch reference.
Produces:
  1. Side-by-side piano roll plots (PNG) for each input
  2. Note-level Precision / Recall / F1 metrics table
  3. Summary statistics

Usage:
    python other/compare_to_bp.py

Outputs go to outputs/comparisons/
"""
import os
import struct
import math
import sys

# ── Minimal MIDI parser (no dependencies beyond stdlib) ──────────────

def parse_midi(path: str) -> list[dict]:
    """Parse a MIDI file, return list of {note, start_sec, end_sec, channel}."""
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != b"MThd":
        raise ValueError(f"Not a MIDI file: {path}")

    # Header
    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, n_tracks, tpq = struct.unpack(">HHH", data[8:14])
    pos = 8 + header_len

    # Collect all note events across all tracks
    all_notes: list[dict] = []
    tempo = 500000  # default 120 BPM

    for _track in range(n_tracks):
        if data[pos:pos+4] != b"MTrk":
            break
        track_len = struct.unpack(">I", data[pos+4:pos+8])[0]
        track_end = pos + 8 + track_len
        pos += 8

        tick = 0
        running_status = 0
        pending: dict[tuple[int,int], tuple[int, int]] = {}  # (ch, note) -> (start_tick, vel)

        while pos < track_end:
            # Variable-length delta
            delta = 0
            while pos < track_end:
                b = data[pos]; pos += 1
                delta = (delta << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            tick += delta

            if pos >= track_end:
                break

            b = data[pos]

            if b == 0xFF:  # Meta event
                pos += 1
                meta_type = data[pos]; pos += 1
                ml = 0
                while pos < track_end:
                    b2 = data[pos]; pos += 1
                    ml = (ml << 7) | (b2 & 0x7F)
                    if not (b2 & 0x80):
                        break
                if meta_type == 0x51 and ml == 3:  # Tempo
                    tempo = (data[pos] << 16) | (data[pos+1] << 8) | data[pos+2]
                pos += ml
                continue

            if b == 0xF0 or b == 0xF7:  # SysEx
                pos += 1
                ml = 0
                while pos < track_end:
                    b2 = data[pos]; pos += 1
                    ml = (ml << 7) | (b2 & 0x7F)
                    if not (b2 & 0x80):
                        break
                pos += ml
                continue

            # Channel message
            if b & 0x80:
                running_status = b
                pos += 1
            status = running_status
            msg_type = status & 0xF0
            ch = status & 0x0F

            if msg_type == 0x90:  # Note On
                note = data[pos]; pos += 1
                vel = data[pos]; pos += 1
                key = (ch, note)
                if vel > 0:
                    pending[key] = (tick, vel)
                else:  # vel=0 means note off
                    if key in pending:
                        start_tick, sv = pending.pop(key)
                        s = start_tick * tempo / (tpq * 1_000_000)
                        e = tick * tempo / (tpq * 1_000_000)
                        all_notes.append(dict(note=note, start_sec=s, end_sec=e, channel=ch, velocity=sv))
            elif msg_type == 0x80:  # Note Off
                note = data[pos]; pos += 1
                pos += 1  # velocity
                key = (ch, note)
                if key in pending:
                    start_tick, sv = pending.pop(key)
                    s = start_tick * tempo / (tpq * 1_000_000)
                    e = tick * tempo / (tpq * 1_000_000)
                    all_notes.append(dict(note=note, start_sec=s, end_sec=e, channel=ch, velocity=sv))
            elif msg_type == 0xC0 or msg_type == 0xD0:
                pos += 1  # 1 data byte
            else:
                pos += 2  # 2 data bytes

        pos = track_end

    all_notes.sort(key=lambda n: n["start_sec"])
    return all_notes


# ── Note matching for metrics ─────────────────────────────────────────

def match_notes(ref: list[dict], est: list[dict],
                onset_tol: float = 0.3, pitch_tol: int = 0) -> tuple[int, int, int]:
    """
    Match estimated notes to reference notes.
    A match requires: pitch within pitch_tol semitones AND onset within onset_tol seconds.
    Returns (true_positives, false_positives, false_negatives).
    """
    ref_matched = [False] * len(ref)
    est_matched = [False] * len(est)

    # Greedy matching: for each est note, find closest unmatched ref note
    for ei, en in enumerate(est):
        best_dist = float("inf")
        best_ri = -1
        for ri, rn in enumerate(ref):
            if ref_matched[ri]:
                continue
            pitch_diff = abs(en["note"] - rn["note"])
            if pitch_diff > pitch_tol:
                continue
            onset_diff = abs(en["start_sec"] - rn["start_sec"])
            if onset_diff > onset_tol:
                continue
            if onset_diff < best_dist:
                best_dist = onset_diff
                best_ri = ri
        if best_ri >= 0:
            ref_matched[best_ri] = True
            est_matched[ei] = True

    tp = sum(est_matched)
    fp = len(est) - tp
    fn = len(ref) - sum(ref_matched)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ── Piano roll plotting ──────────────────────────────────────────────

def plot_comparison(de_notes, bp_notes, title, out_path, duration=None):
    """Plot side-by-side piano rolls: Digital Ear vs Basic Pitch, compact layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if duration is None:
        all_ends = [n["end_sec"] for n in de_notes + bp_notes]
        duration = max(all_ends) if all_ends else 60.0

    # Use TIGHT pitch range — only the overlap region matters
    de_pitches = [n["note"] for n in de_notes] if de_notes else []
    bp_pitches = [n["note"] for n in bp_notes] if bp_notes else []
    all_pitches = de_pitches + bp_pitches
    if not all_pitches:
        return
    # Focus on the Digital Ear range (which is tighter) with some padding
    if de_pitches:
        lo = min(de_pitches) - 4
        hi = max(de_pitches) + 4
    else:
        lo = min(all_pitches) - 2
        hi = max(all_pitches) + 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6), sharex=True, sharey=True,
                                    gridspec_kw={"hspace": 0.08})
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98, color="#e8e0d4")

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    def note_label(n):
        return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"

    for ax, notes, label, color, bar_h in [
        (ax1, de_notes, "Digital Ear", "#39ff14", 0.8),
        (ax2, bp_notes, "Basic Pitch (Spotify)", "#ff6b35", 0.8),
    ]:
        ax.set_facecolor("#0a0a0e")

        # Grid
        for n in range(lo, hi + 1):
            if n % 12 == 0:
                ax.axhline(y=n, color="#252530", linewidth=0.8)
            else:
                ax.axhline(y=n, color="#14141a", linewidth=0.3)

        # Notes
        for n in notes:
            if lo <= n["note"] <= hi:  # clip to visible range
                ax.barh(n["note"], n["end_sec"] - n["start_sec"], left=n["start_sec"],
                        height=bar_h, color=color, alpha=0.9, edgecolor="none")

        ax.set_ylim(lo, hi)
        ax.set_xlim(0, duration)

        # Label on the left inside the plot
        ax.text(duration * 0.005, hi - 1.5, f"{label}  ({len(notes)} notes)",
                fontsize=11, fontweight="bold", color=color, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0e", edgecolor="none", alpha=0.8))

        yticks = [n for n in range(lo, hi + 1) if n % 12 == 0]
        ax.set_yticks(yticks)
        ax.set_yticklabels([note_label(n) for n in yticks], fontsize=9, color="#888")
        ax.tick_params(colors="#555", length=3)
        for spine in ax.spines.values():
            spine.set_color("#333")

    ax2.set_xlabel("Time (seconds)", fontsize=11, color="#aaa")

    fig.patch.set_facecolor("#0a0a0a")
    plt.savefig(out_path, dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_overlay(de_notes, bp_notes, title, out_path, duration=None):
    """Plot overlaid piano rolls: BP thick in back, DE bright on top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if duration is None:
        all_ends = [n["end_sec"] for n in de_notes + bp_notes]
        duration = max(all_ends) if all_ends else 60.0

    # Focus on Digital Ear's pitch range
    de_pitches = [n["note"] for n in de_notes] if de_notes else []
    bp_pitches = [n["note"] for n in bp_notes] if bp_notes else []
    all_pitches = de_pitches + bp_pitches
    if not all_pitches:
        return
    if de_pitches:
        lo = min(de_pitches) - 4
        hi = max(de_pitches) + 4
    else:
        lo = min(all_pitches) - 2
        hi = max(all_pitches) + 2

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    fig.suptitle(f"{title} — Overlay Comparison", fontsize=15, fontweight="bold",
                 y=0.98, color="#e8e0d4")
    ax.set_facecolor("#08080c")

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    def note_label(n):
        return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"

    # Grid
    for n in range(lo, hi + 1):
        if n % 12 == 0:
            ax.axhline(y=n, color="#1e1e28", linewidth=0.8)
        else:
            ax.axhline(y=n, color="#111118", linewidth=0.3)

    # LAYER 1: Basic Pitch — thick, bold orange/red bars in background
    for n in bp_notes:
        if lo <= n["note"] <= hi:
            ax.barh(n["note"], n["end_sec"] - n["start_sec"], left=n["start_sec"],
                    height=0.9, color="#ff6b35", alpha=0.55, edgecolor="none")

    # LAYER 2: Digital Ear — bright neon green, slightly narrower, on top
    for n in de_notes:
        ax.barh(n["note"], n["end_sec"] - n["start_sec"], left=n["start_sec"],
                height=0.6, color="#39ff14", alpha=0.95, edgecolor="#2acc10",
                linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#39ff14", alpha=0.95, edgecolor="#2acc10",
                  label=f"Digital Ear ({len(de_notes)} notes)"),
            Patch(facecolor="#ff6b35", alpha=0.55,
                  label=f"Basic Pitch ({len(bp_notes)} notes)"),
        ],
        loc="upper right", fontsize=11, framealpha=0.9,
        facecolor="#12121a", edgecolor="#444", labelcolor="#ddd",
    )

    ax.set_ylim(lo, hi)
    ax.set_xlim(0, duration)
    yticks = [n for n in range(lo, hi + 1) if n % 12 == 0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([note_label(n) for n in yticks], fontsize=10, color="#999")
    ax.set_xlabel("Time (seconds)", fontsize=12, color="#aaa")
    ax.set_ylabel("Pitch", fontsize=12, color="#aaa")
    ax.tick_params(colors="#666", length=3)
    for spine in ax.spines.values():
        spine.set_color("#333")

    fig.patch.set_facecolor("#0a0a0a")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bp_dir = "/Users/kyle./Documents/GitHub/basic-pitch/output"
    de_dir = os.path.join(project, "outputs")
    out_dir = os.path.join(de_dir, "comparisons")
    os.makedirs(out_dir, exist_ok=True)

    inputs = [
        ("Input 1", "output1_mono.mid", "Output1", "i1_basic_pitch.mid"),
        ("Input 2", "output2_mono.mid", "Output2", "i2_basic_pitch.mid"),
        ("Input 3", "output3_mono.mid", "Output3", "i3_basic_pitch.mid"),
        ("Input 4", "output4_mono.mid", "Output4", "i4_basic_pitch.mid"),
    ]

    print("=" * 70)
    print("  DIGITAL EAR vs BASIC PITCH — Comparison Report")
    print("=" * 70)

    all_metrics = []

    for name, de_file, de_subdir, bp_file in inputs:
        de_path = os.path.join(de_dir, de_subdir, de_file)
        bp_path = os.path.join(bp_dir, bp_file)

        if not os.path.exists(de_path):
            print(f"\n  SKIP {name}: {de_path} not found")
            continue
        if not os.path.exists(bp_path):
            print(f"\n  SKIP {name}: {bp_path} not found")
            continue

        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")

        de_notes = parse_midi(de_path)
        bp_notes = parse_midi(bp_path)

        # For Digital Ear, only use channel 0 (melody)
        de_melody = [n for n in de_notes if n["channel"] == 0]
        # Basic Pitch typically outputs everything on channel 0
        bp_melody = bp_notes

        print(f"  Digital Ear:  {len(de_melody)} melody notes")
        print(f"  Basic Pitch:  {len(bp_melody)} notes")

        # Pitch range
        if de_melody:
            de_lo = min(n["note"] for n in de_melody)
            de_hi = max(n["note"] for n in de_melody)
            print(f"  DE range:     MIDI {de_lo}-{de_hi}")
        if bp_melody:
            bp_lo = min(n["note"] for n in bp_melody)
            bp_hi = max(n["note"] for n in bp_melody)
            print(f"  BP range:     MIDI {bp_lo}-{bp_hi}")

        # Time coverage
        if de_melody:
            de_dur = max(n["end_sec"] for n in de_melody)
            print(f"  DE duration:  {de_dur:.1f}s")
        if bp_melody:
            bp_dur = max(n["end_sec"] for n in bp_melody)
            print(f"  BP duration:  {bp_dur:.1f}s")

        # Metrics: use Basic Pitch as reference
        # Strict: exact pitch match
        tp, fp, fn = match_notes(bp_melody, de_melody, onset_tol=0.3, pitch_tol=0)
        p, r, f1 = prf(tp, fp, fn)
        print(f"\n  Strict match (exact pitch, ±0.3s onset):")
        print(f"    TP={tp}  FP={fp}  FN={fn}")
        print(f"    Precision={p:.1%}  Recall={r:.1%}  F1={f1:.1%}")

        # Tolerant: ±1 semitone
        tp2, fp2, fn2 = match_notes(bp_melody, de_melody, onset_tol=0.3, pitch_tol=1)
        p2, r2, f12 = prf(tp2, fp2, fn2)
        print(f"\n  Tolerant match (±1 semitone, ±0.3s onset):")
        print(f"    TP={tp2}  FP={fp2}  FN={fn2}")
        print(f"    Precision={p2:.1%}  Recall={r2:.1%}  F1={f12:.1%}")

        # Octave-tolerant: pitch class match (note % 12)
        tp3, fp3, fn3 = match_notes(bp_melody, de_melody, onset_tol=0.3, pitch_tol=12)
        p3, r3, f13 = prf(tp3, fp3, fn3)
        print(f"\n  Octave-tolerant match (±12 semitones, ±0.3s onset):")
        print(f"    TP={tp3}  FP={fp3}  FN={fn3}")
        print(f"    Precision={p3:.1%}  Recall={r3:.1%}  F1={f13:.1%}")

        all_metrics.append({
            "name": name,
            "de_notes": len(de_melody),
            "bp_notes": len(bp_melody),
            "strict_p": p, "strict_r": r, "strict_f1": f1,
            "tol_p": p2, "tol_r": r2, "tol_f1": f12,
            "oct_p": p3, "oct_r": r3, "oct_f1": f13,
        })

        # Generate plots
        duration = max(
            max((n["end_sec"] for n in de_melody), default=0),
            max((n["end_sec"] for n in bp_melody), default=0),
        )

        plot_comparison(
            de_melody, bp_melody, name,
            os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_comparison.png"),
            duration=duration,
        )

        plot_overlay(
            de_melody, bp_melody, name,
            os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_overlay.png"),
            duration=duration,
        )

    # Summary table
    if all_metrics:
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY TABLE")
        print(f"{'=' * 70}")
        print(f"  {'Input':<10} {'DE':>5} {'BP':>5}  {'Strict F1':>10} {'±1st F1':>10} {'Oct F1':>10}")
        print(f"  {'─'*10} {'─'*5} {'─'*5}  {'─'*10} {'─'*10} {'─'*10}")
        for m in all_metrics:
            print(f"  {m['name']:<10} {m['de_notes']:>5} {m['bp_notes']:>5}"
                  f"  {m['strict_f1']:>9.1%} {m['tol_f1']:>9.1%} {m['oct_f1']:>9.1%}")

        # Averages
        avg_strict = sum(m["strict_f1"] for m in all_metrics) / len(all_metrics)
        avg_tol = sum(m["tol_f1"] for m in all_metrics) / len(all_metrics)
        avg_oct = sum(m["oct_f1"] for m in all_metrics) / len(all_metrics)
        print(f"  {'─'*10} {'─'*5} {'─'*5}  {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'Average':<10} {'':>5} {'':>5}  {avg_strict:>9.1%} {avg_tol:>9.1%} {avg_oct:>9.1%}")

    print(f"\n  Plots saved to: {out_dir}/")
    print()


if __name__ == "__main__":
    main()
