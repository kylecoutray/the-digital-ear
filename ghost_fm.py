#!/usr/bin/env python3
"""
The Digital Ear — Ghost FM
Headless FM radio -> melody extraction -> synth re-synthesis.

Receives over-the-air FM via RTL-SDR, runs the full Digital Ear
pipeline, and re-synthesizes detected melody as sine tones.

  python ghost_fm.py --freq 89.9M
  python ghost_fm.py --freq 101.1M --output-device 2
  python ghost_fm.py --list-devices

Interactive controls (while running):
  w/s     confidence up/down
  a/d     noise gate up/down
  m       mute/unmute synth
  r       toggle radio passthrough (hear raw FM audio)
  q       quit
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
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


# -- ghost synth (detuned layered oscillators + vibrato) --

class GhostSynth:
    """Ghostly synth — 3 detuned oscillators with slow vibrato.

    Same interface as SineSynth (set_pitch, play_note, stop, generate)
    so it drops right into PipelineRunner.

    Sound: three sines slightly detuned from each other (~4 cents apart),
    slow LFO vibrato on the center pitch, softer attack/release for a
    wavering, haunted music-box feel.
    """

    NOTE_MAX_SEC = 3.0

    # detune offsets in semitones (center, slightly flat, slightly sharp)
    DETUNE = [0.0, -0.04, +0.04]
    # per-oscillator amplitude weights (center louder, sides softer)
    VOICE_AMP = [0.45, 0.30, 0.30]

    # vibrato
    VIB_RATE = 4.5    # Hz — slow wavering
    VIB_DEPTH = 0.003  # semitones of pitch wobble (~5 cents)

    def __init__(self, sr: int = SR):
        self.sr = sr
        self._lock = threading.Lock()
        self.muted: bool = False

        self._freq: float = 0.0
        self._phases: list[float] = [0.0, 0.0, 0.0]
        self._vib_phase: float = 0.0
        self._amplitude: float = 0.0
        self._target_amp: float = 0.0
        self._note_end_sample: int = 0
        self._sample_counter: int = 0

        # slower attack/release than SineSynth for ghostly feel
        self._attack_coeff = 1.0 - math.exp(-2.0 * math.pi * 12.0 / sr)
        self._release_coeff = 1.0 - math.exp(-2.0 * math.pi * 6.0 / sr)

    def set_pitch(self, f0_hz: float):
        with self._lock:
            if f0_hz > 20:
                self._freq = f0_hz
                self._target_amp = 0.55
                self._note_end_sample = self._sample_counter + int(2.0 * self.sr)
            else:
                self._target_amp = 0.0

    def play_note(self, midi_note: int, duration_sec: float):
        duration_sec = min(duration_sec, self.NOTE_MAX_SEC)
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        with self._lock:
            self._freq = freq
            self._target_amp = 0.55
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
            phases = self._phases[:]
            vib_phase = self._vib_phase
            target_amp = self._target_amp
            note_end = self._note_end_sample
            sc = self._sample_counter

        two_pi = 2.0 * math.pi
        vib_inc = two_pi * self.VIB_RATE / self.sr

        for i in range(frames):
            if sc + i >= note_end:
                target_amp = 0.0

            # smooth amplitude
            if target_amp > amp:
                amp += self._attack_coeff * (target_amp - amp)
            else:
                amp += self._release_coeff * (target_amp - amp)

            if amp > 0.001 and freq > 20:
                # vibrato LFO (shared across voices)
                vib = self.VIB_DEPTH * math.sin(vib_phase)
                vib_phase += vib_inc
                if vib_phase > two_pi:
                    vib_phase -= two_pi

                # sum 3 detuned voices
                sample = 0.0
                for v in range(3):
                    # detune in semitones -> freq multiplier
                    detune_semi = self.DETUNE[v] + vib
                    v_freq = freq * (2.0 ** (detune_semi / 12.0))

                    sample += self.VOICE_AMP[v] * math.sin(phases[v])
                    phases[v] += two_pi * v_freq / self.sr
                    if phases[v] > two_pi:
                        phases[v] -= two_pi

                out[i] = amp * sample
            else:
                out[i] = 0.0

        with self._lock:
            self._amplitude = amp
            self._phases = phases
            self._vib_phase = vib_phase
            self._target_amp = target_amp
            self._sample_counter = sc + frames

        return out


# -- pipeline runner (from live_musicbox.py) --

class PipelineRunner:
    """Full Digital Ear pipeline in a background thread."""

    def __init__(
        self,
        audio_q: queue.Queue,
        viz_q: queue.Queue,
        synth: Optional[GhostSynth] = None,
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
    """Reads demodulated FM audio from rtl_fm subprocess."""

    def __init__(self, freq: str, audio_q: queue.Queue,
                 block_size: int = BLOCK_SIZE, squelch: int = 60,
                 gain: Optional[int] = None,
                 radio_q: Optional[queue.Queue] = None):
        self.freq = freq
        self.audio_q = audio_q
        self.radio_q = radio_q  # raw FM audio for passthrough playback
        self.block_size = block_size
        self.squelch = squelch
        self.gain = gain
        self.running = False
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def _build_cmd(self):
        cmd = [
            "rtl_fm",
            "-f", self.freq,
            "-M", "fm",
            "-s", "200k",
            "-r", str(SR),
            "-A", "fast",
            "-l", str(self.squelch),
            "-E", "deemp",
        ]
        if self.gain is not None:
            cmd += ["-g", str(self.gain)]
        cmd.append("-")
        return cmd

    def start(self):
        cmd = self._build_cmd()

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
        bytes_per_block = self.block_size * 2  # 16-bit = 2 bytes/sample
        stream = self._proc.stdout

        while self.running and self._proc.poll() is None:
            raw = stream.read(bytes_per_block)
            if not raw:
                break

            # S16LE -> float32 [-1, 1]
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            if len(samples) < self.block_size:
                padded = np.zeros(self.block_size, dtype=np.float32)
                padded[:len(samples)] = samples
                samples = padded

            try:
                self.audio_q.put_nowait(samples)
            except queue.Full:
                pass

            # feed raw audio for radio passthrough
            if self.radio_q is not None:
                try:
                    self.radio_q.put_nowait(samples.copy())
                except queue.Full:
                    pass

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
        time.sleep(0.3)
        self.freq = freq
        self.start()


# -- keyboard input (non-blocking, raw terminal) --

class KeyReader:
    """Non-blocking single-keypress reader using raw terminal mode."""

    def __init__(self):
        self._old_settings = None

    def start(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

    def stop(self):
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> Optional[str]:
        """Return a key if one is waiting, else None."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# -- main --

def main():
    parser = argparse.ArgumentParser(
        description="The Digital Ear — Ghost FM Synthesizer",
    )
    parser.add_argument(
        "--freq", type=str, default="89.9M",
        help="FM frequency, e.g. 89.9M, 101.1M (default: 89.9M)",
    )
    parser.add_argument(
        "--output-device", type=int, default=None,
        help="Audio output device index for synth (use --list-devices)",
    )
    parser.add_argument(
        "--conf", type=float, default=1.0,
        help="Starting confidence threshold (default: 1.0 = off)",
    )
    parser.add_argument(
        "--rms", type=float, default=0.003,
        help="Starting RMS noise gate (default: 0.003)",
    )
    parser.add_argument(
        "--squelch", type=int, default=60,
        help="rtl_fm squelch level, 0=off (default: 60)",
    )
    parser.add_argument(
        "--gain", type=int, default=None,
        help="rtl_fm gain, omit for auto (try 30-50)",
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
    radio_q: queue.Queue = queue.Queue(maxsize=64)  # raw FM for passthrough

    # synth
    synth = None if args.no_synth else GhostSynth()
    out_stream = None
    radio_mode = [False]  # list so closure can mutate
    radio_buf = [np.zeros(1024, dtype=np.float32)]  # playback buffer

    if synth:
        def output_callback(outdata, frames, time_info, status):
            if radio_mode[0]:
                # drain queue into buffer first
                while True:
                    try:
                        chunk = radio_q.get_nowait()
                        radio_buf[0] = np.concatenate([radio_buf[0], chunk])
                    except queue.Empty:
                        break
                # play from buffer
                buf = radio_buf[0]
                if len(buf) >= frames:
                    outdata[:, 0] = buf[:frames] * 0.5
                    radio_buf[0] = buf[frames:]
                else:
                    outdata[:len(buf), 0] = buf * 0.5
                    outdata[len(buf):, 0] = 0
                    radio_buf[0] = np.zeros(0, dtype=np.float32)
            else:
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

    # FM reader
    fm = FMReader(args.freq, audio_q, squelch=args.squelch, gain=args.gain,
                  radio_q=radio_q)
    fm.start()

    # pipeline
    pipeline = PipelineRunner(audio_q, viz_q, synth=synth,
                              conf_th=args.conf, rms_th=args.rms)
    pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)
    pipeline_thread.start()

    # keyboard reader
    keys = KeyReader()

    # shutdown
    shutting_down = threading.Event()

    def shutdown(signum=None, frame=None):
        if shutting_down.is_set():
            return
        shutting_down.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # print header
    print(f"\n  The Digital Ear — Ghost FM")
    print(f"  Tuned to FM {args.freq}  |  synth {'ON' if synth else 'OFF'}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  w/s  confidence ↑↓    a/d  noise gate ↑↓")
    print(f"  r    radio passthrough    m    mute synth")
    print(f"  q    quit")
    print(f"  ──────────────────────────────────────────────\n")

    note_count = 0
    last_print = 0.0
    muted = False

    keys.start()

    try:
        while not shutting_down.is_set():
            # handle keypresses
            key = keys.get_key()
            if key:
                if key == 'q':
                    break
                elif key == 'w':
                    pipeline.conf_th = min(25.0, pipeline.conf_th + 0.5)
                elif key == 's':
                    pipeline.conf_th = max(1.0, pipeline.conf_th - 0.5)
                elif key == 'd':
                    pipeline.rms_th = min(0.1, pipeline.rms_th + 0.002)
                elif key == 'a':
                    pipeline.rms_th = max(0.001, pipeline.rms_th - 0.002)
                elif key == 'r':
                    radio_mode[0] = not radio_mode[0]
                    # clear radio buffer on toggle for clean switch
                    radio_buf[0] = np.zeros(0, dtype=np.float32)
                    # drain radio queue
                    while not radio_q.empty():
                        try:
                            radio_q.get_nowait()
                        except queue.Empty:
                            break
                elif key == 'm':
                    muted = not muted
                    pipeline.synth_muted = muted
                    if synth:
                        synth.muted = muted

            # drain viz queue
            latest_vf = None
            while True:
                try:
                    vf = viz_q.get_nowait()
                    note_count += len(vf.notes)
                    latest_vf = vf
                except queue.Empty:
                    break

            # status line ~4x/sec
            now = time.monotonic()
            if now - last_print >= 0.25 and latest_vf:
                vf = latest_vf
                if vf.current_midi > 0:
                    note_str = f"{midi_to_name(vf.current_midi):>4s} {vf.current_f0:6.1f} Hz"
                else:
                    note_str = "  --    --.- Hz"

                mute_str = " MUTED" if muted else ""
                mode_str = " [RADIO]" if radio_mode[0] else " [GHOST]"

                line = (
                    f"\r  FM {args.freq:>7s}"
                    f"  |  {note_str}"
                    f"  |  conf {pipeline.conf_th:4.1f}"
                    f"  |  gate {pipeline.rms_th:.3f}"
                    f"  |  {note_count:4d} notes"
                    f"  |  {vf.time_sec:6.1f}s"
                    f"{mode_str}{mute_str}    "
                )
                sys.stdout.write(line)
                sys.stdout.flush()
                last_print = now

            time.sleep(0.05)

    finally:
        keys.stop()
        shutdown()
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
        pipeline_thread.join(timeout=2.0)
        print("\n  Done.\n")


if __name__ == "__main__":
    main()
