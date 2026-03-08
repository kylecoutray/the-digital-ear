#!/usr/bin/env python3
"""
Parameter sweep using clean MIDI as ground truth.

Runs the pipeline with different parameter combinations on the noisy input,
then evaluates each against the clean ground truth MIDI.

Usage:
    python sweep_params.py
"""

import subprocess
import sys
import os
import itertools
import re
import tempfile

# Configuration
NOISY_INPUT = "inputs/i3.m4a"
CLEAN_GT    = "test_outputs/output3_clean.mid"
SWEEP_DIR   = "test_outputs/sweep"

# Parameter grid — CLI-accessible params that affect noisy audio quality
PARAM_GRID = {
    'hp_fc':        [40.0, 60.0, 80.0],       # high-pass cutoff (current: 60)
    'fmin':         [80.0, 100.0],              # min pitch search Hz (current: 80)
    'min_note_sec': [0.10, 0.15, 0.20],         # min note duration (current: 0.15)
    'rms_th':       [0.001, 0.003, 0.005],      # RMS threshold (current: 0.003)
}


def run_pipeline(params: dict, out_path: str) -> bool:
    """Run main.py with given params, return True if successful."""
    cmd = [
        sys.executable, "main.py",
        "--in", NOISY_INPUT,
        "--out", out_path,
        "--hp", str(params['hp_fc']),
        "--fmin", str(params['fmin']),
        "--min-note-sec", str(params['min_note_sec']),
        "--rms-th", str(params['rms_th']),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def evaluate(out_path: str) -> dict | None:
    """Evaluate a MIDI file against ground truth, return metrics."""
    cmd = [
        sys.executable, "evaluate.py",
        "--gt", CLEAN_GT,
        "--ours", out_path,
        "--onset-tol", "0.3",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout

        # Parse the evaluation output
        # Look for: Digital Ear (ours)    245    101   41.2%   41.2%   41.2%
        for line in output.split('\n'):
            if 'Digital Ear' in line:
                parts = line.split()
                # Find numeric values
                nums = []
                for p in parts:
                    p = p.strip('%')
                    try:
                        nums.append(float(p))
                    except ValueError:
                        pass
                if len(nums) >= 5:
                    return {
                        'notes': int(nums[0]),
                        'tp': int(nums[1]),
                        'precision': nums[2],
                        'recall': nums[3],
                        'f1': nums[4],
                    }
        return None
    except subprocess.TimeoutExpired:
        return None


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)

    # Generate all parameter combinations
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combos = list(itertools.product(*values))

    print(f"Parameter sweep: {len(combos)} combinations")
    print(f"Parameters: {keys}")
    print(f"Ground truth: {CLEAN_GT}")
    print(f"Noisy input: {NOISY_INPUT}")
    print()

    results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        tag = "_".join(f"{k}{v}" for k, v in params.items())
        out_path = os.path.join(SWEEP_DIR, f"sweep_{i:03d}.mid")

        # Short description
        desc = " | ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i+1}/{len(combos)}] {desc} ... ", end="", flush=True)

        # Run pipeline
        ok = run_pipeline(params, out_path)
        if not ok:
            print("FAILED")
            continue

        # Evaluate
        metrics = evaluate(out_path)
        if metrics is None:
            print("EVAL FAILED")
            continue

        metrics['params'] = params
        results.append(metrics)
        print(f"F1={metrics['f1']:.1f}%  P={metrics['precision']:.1f}%  R={metrics['recall']:.1f}%  ({metrics['notes']} notes)")

    # Sort by F1
    results.sort(key=lambda r: r['f1'], reverse=True)

    print()
    print("=" * 80)
    print("TOP 10 PARAMETER COMBINATIONS (by F1)")
    print("=" * 80)
    for i, r in enumerate(results[:10]):
        p = r['params']
        print(f"  #{i+1}  F1={r['f1']:5.1f}%  P={r['precision']:5.1f}%  R={r['recall']:5.1f}%  "
              f"notes={r['notes']:3d}  "
              f"hp={p['hp_fc']}  fmin={p['fmin']}  min_note={p['min_note_sec']}  rms={p['rms_th']}")

    print()
    print("=" * 80)
    print("BOTTOM 5 (worst)")
    print("=" * 80)
    for i, r in enumerate(results[-5:]):
        p = r['params']
        print(f"  F1={r['f1']:5.1f}%  P={r['precision']:5.1f}%  R={r['recall']:5.1f}%  "
              f"hp={p['hp_fc']}  fmin={p['fmin']}  min_note={p['min_note_sec']}  rms={p['rms_th']}")

    # Print current defaults for comparison
    current = {'hp_fc': 60.0, 'fmin': 80.0, 'min_note_sec': 0.15, 'rms_th': 0.003}
    for r in results:
        if r['params'] == current:
            print(f"\n>>> CURRENT DEFAULTS: F1={r['f1']:.1f}%  P={r['precision']:.1f}%  R={r['recall']:.1f}%")
            break


if __name__ == '__main__':
    main()
