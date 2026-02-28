#!/usr/bin/env python3
from __future__ import annotations # for forwawrd type references

#import helper modules from digital_ear/
from digital_ear.audio_io import probe_sample_rate, stream_m4a_blocks_ffmpeg
from digital_ear.perf import PerfLogger
from digital_ear.preprocess import Preprocessor, rms
from digital_ear.features import FrameExtractor, hann_window, rfft_mag
from digital_ear.pitch import HPSPitchDetector
from digital_ear.voicing import VoicingGate

import argparse, os, sys, time
from dataclasses import dataclass
import numpy as np 



@dataclass(frozen=True) #immutable class
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

def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(
        prog="the-digital-ear",
        description="Paradromics Digital Ear, streaming audio to MIDI iter. 0 scaffold.",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input audio file (.m4a)")
    p.add_argument("--out", dest="out_path", required=True, help="Output MIDI file (.mid)")
    p.add_argument("--block", dest="block_size", type=int, default=2048, help="Max block size (samples)")
    p.add_argument("--sr", dest = "target_sr", type=int, default=44100, help="Target decode sample rate in Hz. Default 44100.")
    p.add_argument("--dc", dest="dc_fc", type=float, default=30.0, help="DC blocker cutoff Hz")
    p.add_argument("--hp", dest="hp_fc", type=float, default=70.0, help="High-pass cutoff Hz")
    p.add_argument("--lp", dest="lp_fc", type=float, default=1200.0, help="Low-pass cutoff Hz")
    p.add_argument("--nfft", dest="n_fft", type=int, default=2048, help="FFT size (analysis frame length)")
    p.add_argument("--hop", dest="hop", type=int, default=2048, help="Hop size (samples between frames)")
    p.add_argument("--fmin", type=float, default=80.0, help="Min f0 search (Hz)")
    p.add_argument("--fmax", type=float, default=1000.0, help="Max f0 search (Hz)")
    p.add_argument("--conf-th", type=float, default=3.0, help="Confidence threshold for voiced frames")
    p.add_argument("--rms-th", type=float, default=1e-3, help="RMS threshold for voiced frames")
    p.add_argument("--dump-frames", type=str, default="", help="Optional CSV path to dump per-frame (debug)")
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

    # instantiate the preprocessor
    pre = Preprocessor(fs=float(out_sr), dc_fc=args.dc_fc, hp_fc=args.hp_fc, lp_fc=args.lp_fc)

    # iteration 3 addition: frame extractor + window
    fx = FrameExtractor(n_fft=args.n_fft, hop=args.hop)
    win = hann_window(args.n_fft)
    det = HPSPitchDetector(sr=float(out_sr), n_fft=args.n_fft, fmin=args.fmin, fmax=args.fmax)

    gate = VoicingGate(conf_threshold=args.conf_th, rms_threshold=args.rms_th)
    stream_sample_index = 0  # absolute index of current block start in decoded stream

    csv_f = None
    if args.dump_frames:
        csv_f = open(args.dump_frames, "w", encoding="utf-8")
        csv_f.write("frame_idx,time_sec,f0_hz,conf,rms,voiced\n")

    frame_count = 0

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

        #Debug stats before preprocessing, only for first 5 blocks to avoid huge logs.
        if args.debug and block_count <= 5:
            mean_in = float(np.mean(blk)) if blk.size else 0.0
            rms_in = rms(blk)

        # begin preprocessing (iteration 2)
        y = pre.process(blk)

        #debug stats after preprocessing
        if args.debug and block_count <= 5: #only first 5 blocks to avoid huge logs
            mean_out = float(np.mean(y)) if y.size else 0.0
            rms_out = rms(y)
            print(f"DBG_BLOCK={block_count} N={n} MEAN_IN={mean_in:.6e} MEAN_OUT={mean_out:.6e} RMS_IN={rms_in:.6f} RMS_OUT={rms_out:.6f}")


        #iteration 3: push into frame extractor and computer FFT mag for each frame
        for frame, frame_start in fx.push_indexed(y, stream_sample_index):
            mag = rfft_mag(frame, win)
            f0, conf = det.estimate(mag)

            frame_rms = rms(frame)
            f0_gated, voiced = gate.apply(f0, conf, frame_rms)

            t_sec = frame_start / float(out_sr)

            if args.debug and frame_count < 20:
                if voiced:
                    print(f"DBG_VOICE frame={frame_count} t={t_sec:.3f} f0={f0_gated:.2f} conf={conf:.2f} rms={frame_rms:.4f}")
                else:
                    print(f"DBG_VOICE frame={frame_count} t={t_sec:.3f} f0=None conf={conf:.2f} rms={frame_rms:.4f}")

            if csv_f is not None:
                f0_val = "" if f0_gated is None else f"{f0_gated:.6f}"
                csv_f.write(f"{frame_count},{t_sec:.6f},{f0_val},{conf:.6f},{frame_rms:.6f},{int(voiced)}\n")

            frame_count += 1
        
        stream_sample_index += n #update once per block 



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
    print(f"FRAMES_PROCESSED={frame_count}")

    print(f"ELAPSED_SEC={elapsed_s:.6f}")
    print(f"PEAK_RSS_MB={peak_mb:.2f}")

    if args.debug:
        print("STATUS=OK (this is where future iterations will report more detailed status)")

    if csv_f is not None:
        csv_f.close()

    return 0

# guard for imports 
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))