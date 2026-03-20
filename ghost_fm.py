#!/usr/bin/env python3
"""
The Digital Ear — Ghost FM
Headless FM radio → melody extraction → synth re-synthesis.

Receives over-the-air FM via RTL-SDR, runs the full Digital Ear
pipeline, and re-synthesizes detected melody as sine tones.

  python ghost_fm.py --freq 99.5M
  python ghost_fm.py --freq 101.1M --output-device 2
  python ghost_fm.py --list-devices
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd

from digital_ear.preprocess import Preprocessor, rms
from digital_ear.hpss import HPSS
from digital_ear.features import FrameExtractor
from digital_ear.harmonic_pitch import HarmonicPitchDetector
from digital_ear.melody_extractor import MelodyExtractor
from digital_ear.note_tracker import NoteTracker, NoteEvent


# -- constants (match live_musicbox / main.py) --

SR = 44100
BLOCK_SIZE = 2048
N_FFT = 2048
HOP = 512

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


@dataclass
class VizFrame:
    """One frame of pipeline output for status display."""
    time_sec: float = 0.0
    audio_time: float = 0.0
    rms_level: float = 0.0
    current_f0: float = 0.0
    current_midi: int = 0
    confidence: float = 0.0
    notes: list = field(default_factory=list)


# -- sine synth (copied from live_musicbox.py) --

class SineSynth:
    """Thread-safe sine synth with pitch-following and 2s safety timeout."""

    NOTE_MAX_SEC = 3.0

    def __init__(self, sr: int = SR):
        self.sr = sr
        self._lock = threading.Lock()
        self.muted: bool = False

        self._freq: float = 0.0
        self._phase: float = 0.0
        self._amplitude: float = 0.0
        self._target_amp: float = 0.0
        self._note_end_sample: int = 0
        self._sample_counter: int = 0

        self._attack_coeff = 1.0 - math.exp(-2.0 * math.pi * 30.0 / sr)
        self._release_coeff = 1.0 - math.exp(-2.0 * math.pi * 20.0 / sr)

    def set_pitch(self, f0_hz: float):
        with self._lock:
            if f0_hz > 20:
                self._freq = f0_hz
                self._target_amp = 0.30
                self._note_end_sample = self._sample_counter + int(2.0 * self.sr)
            else:
                self._target_amp = 0.0

    def play_note(self, midi_note: int, duration_sec: float):
        duration_sec = min(duration_sec, self.NOTE_MAX_SEC)
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        with self._lock:
            self._freq = freq
            self._target_amp = 0.30
            self._note_end_sample = self._sample_counter + int(duration_sec * self.sr)

    def stop(self):
        with self._lock:
            self._target_amp = 0.0

    def generate(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)

        if self.muted:
            with self._lock:
                self._amplitude = 0.0
                self._target_amp = 0.0
                self._sample_counter += frames
            return out

        with self._lock:
            freq = self._freq
            amp = self._amplitude
            phase = self._phase
            target_amp = self._target_amp
            note_end = self._note_end_sample
            sc = self._sample_counter

        for i in range(frames):
            if sc + i >= note_end:
                target_amp = 0.0
            if target_amp > amp:
                amp += self._attack_coeff * (target_amp - amp)
            else:
                amp += self._release_coeff * (target_amp - amp)

            if amp > 0.001 and freq > 20:
                out[i] = amp * math.sin(phase)
                phase += 2.0 * math.pi * freq / self.sr
                if phase > 2.0 * math.pi:
                    phase -= 2.0 * math.pi
            else:
                out[i] = 0.0

        with self._lock:
            self._amplitude = amp
            self._phase = phase
            self._target_amp = target_amp
            self._sample_counter = sc + frames

        return out


# -- pipeline runner (copied from live_musicbox.py) --

class PipelineRunner:
    """Full Digital Ear pipeline in a background thread."""

    def __init__(
        self,
        audio_q: queue.Queue,
        viz_q: queue.Queue,
        synth: Optional[SineSynth] = None,
        fmin: float = 80.0,
        fmax: float = 1000.0,
        conf_th: float = 1.0,
        rms_th: float = 0.003,
    ):
        self.audio_q = audio_q
        self.viz_q = viz_q
        self.synth = synth
        self.running = True

        self.conf_th: float = conf_th
        self.rms_th: float = rms_th
        self.synth_muted: bool = False

        hop_sec = HOP / float(SR)
        self.pre = Preprocessor(fs=float(SR), dc_fc=30.0, hp_fc=60.0, lp_fc=4000.0)
        self.hpss = HPSS(n_fft=N_FFT, hop=HOP, kernel_h=31, kernel_p=31, power=2.0)
        self.fx = FrameExtractor(n_fft=N_FFT, hop=HOP)
        self.det = HarmonicPitchDetector(
            sr=float(SR), n_fft=N_FFT, hop=HOP,
            fmin=fmin, fmax=fmax,
            conf_threshold=7.0,  # only used by estimate(), not our path
        )
        hpss_delay_sec = (self.hpss.kernel_h // 2) * hop_sec
        latency_sec = N_FFT / (2.0 * float(SR)) - hpss_delay_sec

        self.melody = MelodyExtractor(hop_sec=hop_sec)
        self.tracker = NoteTracker(
            hop_sec=hop_sec, median_window=15,
            min_note_sec=0.15, merge_gap_sec=0.12,
            latency_sec=latency_sec,
        )

        self._stream_sample_idx = 0
        self._start_time = 0.0
        self._latest_rms = 0.0
        self._latest_f0 = 0.0
        self._latest_midi = 0
        self._latest_conf = 0.0
        self._silent_frames = 0

    def run(self):
        self._start_time = time.monotonic()

        while self.running:
            try:
                blk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if blk is None:
                break

            new_notes: list[NoteEvent] = []
            wall_sec = time.monotonic() - self._start_time

            y = self.pre.process(blk)

            self._latest_rms = rms(y)
            if self._latest_rms < self.rms_th * 0.5 and self.synth and not self.synth_muted:
                self.synth.set_pitch(0)

            self.hpss.push(y)

            for harmonic_block in self.hpss.pop_harmonic():
                for frame, frame_start in self.fx.push_indexed(
                    harmonic_block, self._stream_sample_idx
                ):
                    frame_rms = rms(frame)
                    rms_th = self.rms_th
                    conf_th = self.conf_th

                    if frame_rms < rms_th:
                        candidates = []
                        tonality = 0.0
                        self._silent_frames += 1
                    else:
                        candidates, tonality = self.det.estimate_candidates(
                            frame, n_top=5, min_conf=1.5
                        )
                        if conf_th > 1.5:
                            candidates = [(hz, c) for hz, c in candidates
                                          if c >= conf_th]
                        self._silent_frames = 0

                    f0_hz = self.melody.push(candidates, tonality)

                    if self._silent_frames > 8:
                        if self.synth and not self.synth_muted:
                            self.synth.set_pitch(0)

                    if f0_hz is not None:
                        self._latest_f0 = f0_hz
                        if f0_hz > 0:
                            self._latest_midi = int(round(69 + 12 * math.log2(f0_hz / 440.0)))
                            self._latest_conf = candidates[0][1] if candidates else 0
                        else:
                            self._latest_midi = 0
                            self._latest_conf = 0

                        if self.synth and not self.synth_muted:
                            self.synth.set_pitch(f0_hz if f0_hz > 0 else 0)
                        elif self.synth and self.synth_muted:
                            self.synth.set_pitch(0)

                        tracker_f0 = f0_hz if f0_hz > 0 else None
                        events = self.tracker.push(tracker_f0)
                        new_notes.extend(events)

                    elif not candidates:
                        self._latest_midi = 0
                        self._latest_f0 = 0.0
                        self._latest_conf = 0.0
                        if self.synth and not self.synth_muted:
                            self.synth.set_pitch(0)

                self._stream_sample_idx += len(harmonic_block)

            audio_time = self._stream_sample_idx / float(SR)
            vf = VizFrame(
                time_sec=wall_sec,
                audio_time=audio_time,
                rms_level=self._latest_rms,
                current_f0=self._latest_f0,
                current_midi=self._latest_midi,
                confidence=self._latest_conf,
                notes=new_notes,
            )
            try:
                self.viz_q.put_nowait(vf)
            except queue.Full:
                pass

    def stop(self):
        self.running = False


# -- FM radio reader --

class FMReader:
    """Reads demodulated FM audio from rtl_fm subprocess.

    Spawns rtl_fm, reads signed 16-bit LE PCM from stdout,
    converts to float32 numpy arrays, pushes to audio queue.
    """

    def __init__(self, freq: str, audio_q: queue.Queue, block_size: int = BLOCK_SIZE):
        self.freq = freq
        self.audio_q = audio_q
        self.block_size = block_size
        self.running = False
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        cmd = [
            "rtl_fm",
            "-f", self.freq,
            "-M", "fm",
            "-s", "170k",
            "-r", str(SR),
            "-A", "fast",
            "-l", "0",
            "-",
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("ERROR: rtl_fm not found. Install with: sudo apt install rtl-sdr")
            sys.exit(1)

        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        bytes_per_block = self.block_size * 2  # 16-bit = 2 bytes per sample
        stream = self._proc.stdout

        while self.running and self._proc.poll() is None:
            raw = stream.read(bytes_per_block)
            if not raw:
                break

            # S16LE -> float32 in [-1, 1]
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # handle partial reads
            if len(samples) < self.block_size:
                padded = np.zeros(self.block_size, dtype=np.float32)
                padded[:len(samples)] = samples
                samples = padded

            try:
                self.audio_q.put_nowait(samples)
            except queue.Full:
                pass  # pipeline behind, drop block

    def stop(self):
        self.running = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def set_freq(self, freq: str):
        """Change frequency by restarting rtl_fm."""
        self.stop()
        self.freq = freq
        self.start()


# -- main --

def main():
    parser = argparse.ArgumentParser(
        description="The Digital Ear — Ghost FM Synthesizer",
    )
    parser.add_argument(
        "--freq", type=str, default="99.5M",
        help="FM frequency, e.g. 99.5M, 101.1M (default: 99.5M)",
    )
    parser.add_argument(
        "--output-device", type=int, default=None,
        help="Audio output device index for synth (use --list-devices)",
    )
    parser.add_argument(
        "--no-synth", action="store_true",
        help="Disable synth output (detection only)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List audio output devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # queues
    audio_q: queue.Queue = queue.Queue(maxsize=64)
    viz_q: queue.Queue = queue.Queue(maxsize=128)

    # synth
    synth = None if args.no_synth else SineSynth()
    out_stream = None

    # start synth output stream
    if synth:
        def output_callback(outdata, frames, time_info, status):
            outdata[:, 0] = synth.generate(frames)

        try:
            out_stream = sd.OutputStream(
                samplerate=SR,
                channels=1,
                blocksize=1024,
                dtype="float32",
                device=args.output_device,
                callback=output_callback,
            )
            out_stream.start()
        except Exception as e:
            print(f"Audio output error: {e}")
            print("Try --list-devices to find the right output device index")
            sys.exit(1)

    # start FM reader
    fm = FMReader(args.freq, audio_q)
    fm.start()
    print(f"Tuned to FM {args.freq} — receiving...")

    # start pipeline
    pipeline = PipelineRunner(audio_q, viz_q, synth=synth)
    pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)
    pipeline_thread.start()

    # clean shutdown
    shutting_down = threading.Event()

    def shutdown(signum=None, frame=None):
        if shutting_down.is_set():
            return
        shutting_down.set()
        print("\nShutting down...")
        pipeline.stop()
        fm.stop()
        if out_stream:
            try:
                out_stream.stop()
                out_stream.close()
            except Exception:
                pass
        try:
            audio_q.put_nowait(None)
        except queue.Full:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # status display loop
    note_count = 0
    last_print = 0.0

    print(f"Ghost FM running — synth {'ON' if synth else 'OFF'}")
    print("Press Ctrl+C to stop\n")

    try:
        while not shutting_down.is_set():
            # drain viz queue
            latest_vf = None
            while True:
                try:
                    vf = viz_q.get_nowait()
                    note_count += len(vf.notes)
                    latest_vf = vf
                except queue.Empty:
                    break

            # print status ~4x/sec
            now = time.monotonic()
            if now - last_print >= 0.25 and latest_vf:
                vf = latest_vf
                if vf.current_midi > 0:
                    note_str = f"{midi_to_name(vf.current_midi):>4s} {vf.current_f0:6.1f} Hz"
                else:
                    note_str = "  --    --.- Hz"

                line = (
                    f"\r  FM {args.freq:>7s}"
                    f"  |  {note_str}"
                    f"  |  RMS {vf.rms_level:.4f}"
                    f"  |  {note_count:4d} notes"
                    f"  |  {vf.time_sec:6.1f}s"
                )
                sys.stdout.write(line)
                sys.stdout.flush()
                last_print = now

            time.sleep(0.05)

    except KeyboardInterrupt:
        shutdown()

    # wait for pipeline thread
    pipeline_thread.join(timeout=2.0)
    print("\nDone.")


if __name__ == "__main__":
    main()
