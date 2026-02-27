#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, sys, time
from dataclasses import dataclass

import psutil #read process memory

@dataclass(frozen=True) #immutable class
class Args:
    in_path: str
    out_path: str
    block_size: int
    verbose: bool

def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(
        prog="the-digital-ear",
        description="Paradromics Digital Ear, streaming audio to MIDI iter. 0 scaffold.",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input audio file (.m4a)")
    p.add_argument("--out", dest="out_path", required=True, help="Output MIDI file (.mid)")
    p.add_argument("--block", dest="block_size", type=int, default=2048, help="Max block size (samples)")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info")

    ns = p.parse_args(argv)

    # added failsafe checks for inpts
    if ns.block_size <= 0:
        raise SystemExit("ERROR: --block must be positive.")
    if ns.block_size > 2048:
        raise SystemExit("ERROR: --block must be <= 2048.")
    if not os.path.exists(ns.in_path):
        raise SystemExit(f"ERROR: input file not found: {ns.in_path}")
    if not ns.out_path.lower().endswith(".mid"):
        raise SystemExit("ERROR: --out must end with .mid")
    
    return Args(
        in_path=ns.in_path,
        out_path=ns.out_path,
        block_size=ns.block_size,
        verbose=ns.verbose,
    )

class PerfLogger:
    """
    Tracks time and peak resident memory, RSS, in MB. 
    Ensure we are under safety constraints for Pi.
    """

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self._t0 = 0.0
        self._peak_rss = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._sample_rss()

    def _sample_rss(self) -> None:
        rss = self._proc.memory_info().rss  # bytes
        if rss > self._peak_rss:
            self._peak_rss = rss #storing peak rss

    def stop_and_report(self) -> tuple[float, float]:
        self._sample_rss()
        elapsed_s = time.perf_counter() - self._t0
        peak_mb = self._peak_rss / (1024 * 1024)
        return elapsed_s, peak_mb
    
def touch_output_file(path: str) -> None:
    """
    Iteration 0: no MIDI yet.
    We create/truncate the output file so the CLI feels real.
    """
    with open(path, "wb") as f:
        f.write(b"")  # placeholder

def main(argv: list[str]) -> int:
    args = parse_args(argv)

    perf = PerfLogger()
    perf.start()

    # Iteration 0 placeholder work: validate arguments and create output stub.
    touch_output_file(args.out_path)

    elapsed_s, peak_mb = perf.stop_and_report()

    # Consistent, parseable log lines for the deliverable
    # Keep these stable across later iterations.
    print(f"INPUT_FILE={args.in_path}")
    print(f"OUTPUT_FILE={args.out_path}")
    print(f"BLOCK_SIZE={args.block_size}")
    print(f"ELAPSED_SEC={elapsed_s:.6f}")
    print(f"PEAK_RSS_MB={peak_mb:.2f}")

    if args.verbose:
        print("STATUS=OK (Iteration 0 scaffolding only)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))