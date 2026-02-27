#!/usr/bin/env python3
from __future__ import annotations # for forwawrd type references... good practice
from src.audio_io import probe_sample_rate, stream_m4a_blocks_ffmpeg
from src.perf import PerfLogger

import argparse, os, sys, time
from dataclasses import dataclass

import psutil #read process memory
import numpy as np 
import subprocess
import json


@dataclass(frozen=True) #immutable class
class Args:
    in_path: str
    out_path: str
    block_size: int
    target_sr: int
    debug: bool

def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(
        prog="the-digital-ear",
        description="Paradromics Digital Ear, streaming audio to MIDI iter. 0 scaffold.",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input audio file (.m4a)")
    p.add_argument("--out", dest="out_path", required=True, help="Output MIDI file (.mid)")
    p.add_argument("--block", dest="block_size", type=int, default=2048, help="Max block size (samples)")
    p.add_argument("--sr", dest = "target_sr", type=int, default=44100, help="Target decode sample rate in Hz. Default 44100.")
    p.add_argument("--debug", action="store_true", help="Print extra debug info")

    ns = p.parse_args(argv)

    # added failsafe checks for inpts
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
    
    return Args(
        in_path=ns.in_path,
        out_path=ns.out_path,
        block_size=ns.block_size,
        target_sr=ns.target_sr,
        debug=ns.debug,
    )




def touch_output_file(path: str) -> None:
    """
    Iteration 1: still no MIDI yet.
    We create/truncate the output file so the CLI feels real.
    """
    with open(path, "wb") as f:
        f.write(b"")  # placeholder

def main(argv: list[str]) -> int:
    args = parse_args(argv) # parse args

    # start RSS (MB) and time logger
    perf = PerfLogger() 
    perf.start()

    input_sr = probe_sample_rate(args.in_path)
    out_sr, blocks = stream_m4a_blocks_ffmpeg(args.in_path, args.block_size, target_sr=args.target_sr)

    block_count = 0
    max_len_seen = 0
    total_samples = 0

    for blk in blocks:
        perf.sample()  # updates peak RSS as we stream
        block_count += 1
        n = int(blk.shape[0])
        total_samples += n
        if n > max_len_seen:
            max_len_seen = n

        # Acceptance check
        assert n <= args.block_size, f"Block too large: {n} > {args.block_size}"

        #no DSP yet for this stage! nothing w/ blk beyond this point.

    # Still create a placeholder output file for now
    touch_output_file(args.out_path)

    elapsed_s, peak_mb = perf.stop_and_report()

    # Consistent, parseable log lines for the deliverable
    # Keep these stable across later iterations!
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

    print(f"ELAPSED_SEC={elapsed_s:.6f}")
    print(f"PEAK_RSS_MB={peak_mb:.2f}")

    if args.debug:
        print("STATUS=OK (this is where future iterations will report more detailed status)")

    return 0

# guard for imports 
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))