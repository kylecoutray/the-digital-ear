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

from digital_ear.preprocess import Preprocessor, rms
from digital_ear.hpss import HPSS
from digital_ear.features import FrameExtractor
from digital_ear.harmonic_pitch import HarmonicPitchDetector
from digital_ear.melody_extractor import MelodyExtractor
from digital_ear.note_tracker import NoteTracker, NoteEvent
from ghost_display import GhostDisplay


# constants (match live_musicbox / main.py)
SR = 44100
BLOCK_SIZE = 2048
N_FFT = 2048
HOP = 512

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except OSError as e:
        raise RuntimeError(
            "sounddevice could not load PortAudio. Install it with: "
            "sudo apt install -y libportaudio2"
        ) from e


def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def add_display_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--display-backend",
        choices=["st7789", "fbdev", "none"],
        default="st7789",
        help="Display output backend (default: st7789)",
    )
    parser.add_argument(
        "--fbdev",
        default="/dev/fb1",
        help="Framebuffer device for --display-backend fbdev (default: /dev/fb1)",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=240,
        help="Display width in pixels (default: 240; use 480 for MHS35)",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=240,
        help="Display height in pixels (default: 240; use 320 for MHS35)",
    )
    parser.add_argument(
        "--display-rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Software rotation for framebuffer output (default: 0)",
    )
    parser.add_argument(
        "--display-byte-order",
        choices=["little", "big"],
        default="little",
        help="RGB565 byte order for framebuffer output (default: little)",
    )
    parser.add_argument(
        "--display-normal-assets",
        action="store_true",
        help="Use the less saturated normal GhostFM assets",
    )
    parser.add_argument(
        "--no-joystick",
        action="store_true",
        help="Disable GPIO joystick/HAT button input",
    )
    parser.add_argument(
        "--display-test",
        action="store_true",
        help="Run an animated display test without SDR/audio/pipeline startup",
    )


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


def make_display(args) -> GhostDisplay:
    return GhostDisplay(
        backlight_pct=args.brightness,
        backend=args.display_backend,
        fbdev=args.fbdev,
        width=args.display_width,
        height=args.display_height,
        rotation=args.display_rotate,
        byte_order=args.display_byte_order,
        normal_assets=args.display_normal_assets or args.display_backend == "fbdev",
    )


def run_display_test(args):
    lcd = make_display(args)
    lcd.fm_freq = args.freq
    lcd.paused = False
    lcd.conf_th = args.conf
    lcd.rms_th = args.rms
    lcd.mode = "GHOST"
    lcd.start()

    if args.display_backend == "none":
        print("  Display test running with --display-backend none; frames render but are not written.")
    print("  Display test running. Press Ctrl-C to stop.")

    notes = [60, 64, 67, 72, 76, 79, 84, 79, 76, 72, 67, 64]
    try:
        i = 0
        while True:
            midi = notes[i % len(notes)]
            lcd.note_name = midi_to_name(midi)
            lcd.freq_hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            lcd.note_count = i
            lcd.conf_th = args.conf + ((i % 8) * 0.5)
            lcd.rms_th = args.rms + ((i % 5) * 0.001)
            lcd.muted = (i % 10) >= 8
            lcd.mode = "RADIO" if (i % 12) >= 9 else "GHOST"
            lcd.push_note(midi)
            i += 1
            time.sleep(0.45)
    except KeyboardInterrupt:
        pass
    finally:
        lcd.stop()
        print("\n  Display test stopped.\n")


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

        # match live_musicbox envelope speeds
        self._attack_coeff = 1.0 - math.exp(-2.0 * math.pi * 30.0 / sr)
        self._release_coeff = 1.0 - math.exp(-2.0 * math.pi * 20.0 / sr)

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
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except (termios.error, OSError):
            # no terminal (e.g., running under systemd)
            self._old_settings = None

    def stop(self):
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> Optional[str]:
        """Return a key if one is waiting, else None."""
        if self._old_settings is None:
            return None  # no terminal
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except (OSError, ValueError):
            pass
        return None


# -- joystick input (GPIO, Waveshare 1.3" LCD HAT, mounted upside-down) --

class JoystickReader:
    """Reads joystick on Waveshare 1.3" LCD HAT via GPIO.

    Mounted upside-down, so directions are flipped:
      physical UP   (GPIO 6)  -> 's'  (conf down)
      physical DOWN (GPIO 19) -> 'w'  (conf up)
      physical LEFT (GPIO 5)  -> 'd'  (gate up)
      physical RIGHT(GPIO 26) -> 'a'  (gate down)

    Joystick center press: short = mute, hold 1s = reset defaults.
    """

    # GPIO pin -> key mapping (flipped for upside-down mount)
    # Hold-detect pins (13, 21) handled separately below
    PIN_MAP = {
        6:  's',   # physical UP    -> conf down
        19: 'w',   # physical DOWN  -> conf up
        5:  'd',   # physical LEFT  -> gate up
        26: 'a',   # physical RIGHT -> gate down
        20: 'r',   # KEY2 (middle)                  -> radio toggle
    }

    # Hold-detect pins: (gpio, threshold_sec, short_key, hold_key)
    HOLD_PINS = [
        (13, 1.0, 'm', 'x'),   # joystick press: short=mute, hold=reset
        (16, 1.5, 'f', 'e'),   # KEY3: short=cycle preset, hold 1.5s=edit freq
        (21, 3.0, 'p', 'Q'),   # KEY1 (bottom): short=pause, hold=quit
    ]

    def __init__(self):
        self._buttons = {}
        self._prev_state = {}
        self._available = False
        # hold state per pin: {pin: (press_time, was_pressed, fired)}
        self._hold_state: dict[int, list] = {}

    def start(self):
        try:
            from gpiozero import Button as GpioButton
            all_pins = list(self.PIN_MAP.keys()) + [h[0] for h in self.HOLD_PINS]
            for pin in all_pins:
                btn = GpioButton(pin, pull_up=True, bounce_time=0.05)
                self._buttons[pin] = btn
                self._prev_state[pin] = False
            for pin, _, _, _ in self.HOLD_PINS:
                self._hold_state[pin] = [0.0, False, False]  # press_time, was_pressed, fired
            self._available = True
        except Exception as e:
            print(f"  Joystick not available: {e}")
            self._available = False

    def get_key(self) -> Optional[str]:
        """Return mapped key on new press (edge-triggered), else None.

        Hold-detect pins return short_key on release (if < threshold)
        or hold_key when threshold is reached (while still pressed).
        Joystick center: short = mute, hold 3s = edit mode.
        """
        if not self._available:
            return None

        # check regular buttons first
        for pin, btn in self._buttons.items():
            if pin in self._hold_state:
                continue
            pressed = btn.is_pressed
            if pressed and not self._prev_state[pin]:
                self._prev_state[pin] = True
                return self.PIN_MAP[pin]
            elif not pressed:
                self._prev_state[pin] = False

        # handle hold-detect pins
        for pin, threshold, short_key, hold_key in self.HOLD_PINS:
            if pin not in self._buttons:
                continue
            pressed = self._buttons[pin].is_pressed
            state = self._hold_state[pin]  # [press_time, was_pressed, fired]

            if pressed and not state[1]:
                # just pressed
                state[1] = True
                state[0] = time.monotonic()
                state[2] = False
            elif pressed and state[1] and not state[2]:
                # still holding — check threshold
                if time.monotonic() - state[0] >= threshold:
                    state[2] = True
                    return hold_key
            elif not pressed and state[1]:
                # just released
                state[1] = False
                if not state[2]:
                    return short_key
                state[2] = False

        return None

    def stop(self):
        for btn in self._buttons.values():
            try:
                btn.close()
            except Exception:
                pass
        self._buttons.clear()


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
    parser.add_argument(
        "--brightness", type=int, default=50,
        help="LCD backlight brightness 0-100 (default: 50)",
    )
    add_display_args(parser)
    args = parser.parse_args()

    if args.display_test:
        run_display_test(args)
        return

    sd = load_sounddevice()

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
    muted_ref = [False]  # list so audio callback can read it

    if synth:
        def output_callback(outdata, frames, time_info, status):
            if muted_ref[0]:
                outdata[:, 0] = 0
                return
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

    # FM presets
    FM_PRESETS = ["89.5M", "89.1M", "91.3M", "89.9M"]
    current_preset = [0]  # list so closure can mutate
    # find starting preset index
    for i, p in enumerate(FM_PRESETS):
        if p == args.freq:
            current_preset[0] = i
            break

    # FM reader (not started yet — starts when unpaused)
    fm = FMReader(args.freq, audio_q, squelch=args.squelch, gain=args.gain,
                  radio_q=radio_q)

    # pipeline
    pipeline = PipelineRunner(audio_q, viz_q, synth=synth,
                              conf_th=args.conf, rms_th=args.rms)
    pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)
    pipeline_thread.start()

    # keyboard reader
    keys = KeyReader()
    joy = JoystickReader()

    # LCD display
    lcd = make_display(args)
    lcd.fm_freq = args.freq
    lcd.paused = True  # start in idle mode

    # paused state — FM reader not started yet
    is_paused = [True]

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
    print(f"  e    edit frequency       f    cycle preset")
    print(f"  q    quit")
    print(f"  ──────────────────────────────────────────────\n")

    note_count = 0
    last_print = 0.0
    muted = False

    # manual frequency edit mode (triple-click joystick center)
    edit_mode = [False]
    # store freq as tenths of MHz: 89.5 -> 895, 101.1 -> 1011
    def freq_to_tenths(f: str) -> int:
        return int(round(float(f.rstrip('Mm')) * 10))
    def tenths_to_freq(t: int) -> str:
        return f"{t / 10:.1f}M"
    edit_tenths = [freq_to_tenths(args.freq)]
    edit_cursor = [0]  # 0=tens, 1=ones, 2=tenths
    EDIT_STEPS = [100, 10, 1]  # increment per cursor position (in tenths)

    keys.start()
    if not args.no_joystick:
        joy.start()
    lcd.start()

    try:
        while not shutting_down.is_set():
            # handle keypresses (keyboard or joystick)
            key = keys.get_key() or joy.get_key()
            if key:
                # ---- EDIT MODE: joystick remapped to freq digit editing ----
                if edit_mode[0]:
                    if key == 'e':
                        # confirm & exit edit mode
                        edit_mode[0] = False
                        lcd.edit_mode = False
                        new_freq = tenths_to_freq(edit_tenths[0])
                        lcd.fm_freq = new_freq
                        if not is_paused[0]:
                            fm.set_freq(new_freq)
                        else:
                            fm.freq = new_freq
                        print(f"\r  ** EDIT CONFIRMED — FM {new_freq} **                 ")
                    elif key == 'w':
                        # joystick up → increment digit
                        edit_tenths[0] = min(1080, edit_tenths[0] + EDIT_STEPS[edit_cursor[0]])
                        lcd.edit_freq_str = tenths_to_freq(edit_tenths[0]).rstrip('M')
                    elif key == 's':
                        # joystick down → decrement digit
                        edit_tenths[0] = max(875, edit_tenths[0] - EDIT_STEPS[edit_cursor[0]])
                        lcd.edit_freq_str = tenths_to_freq(edit_tenths[0]).rstrip('M')
                    elif key == 'a':
                        # joystick left → move cursor left
                        edit_cursor[0] = max(0, edit_cursor[0] - 1)
                        lcd.edit_cursor = edit_cursor[0]
                    elif key == 'd':
                        # joystick right → move cursor right
                        edit_cursor[0] = min(2, edit_cursor[0] + 1)
                        lcd.edit_cursor = edit_cursor[0]
                    elif key == 'm':
                        # short press → cancel edit, restore original freq
                        edit_mode[0] = False
                        lcd.edit_mode = False
                        print(f"\r  ** EDIT CANCELLED **                                  ")
                    elif key == 'q':
                        break
                    # ignore other keys during edit mode
                    continue

                # ---- NORMAL MODE ----
                if key == 'q':
                    break
                elif key == 'e':
                    # enter edit mode (hold joystick center 3s)
                    edit_mode[0] = True
                    edit_tenths[0] = freq_to_tenths(fm.freq)
                    edit_cursor[0] = 2  # start on tenths digit (finest control)
                    lcd.edit_mode = True
                    lcd.edit_freq_str = tenths_to_freq(edit_tenths[0]).rstrip('M')
                    lcd.edit_cursor = edit_cursor[0]
                    print(f"\r  ** EDIT MODE — use joystick to change freq **         ")
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
                    muted_ref[0] = muted
                    pipeline.synth_muted = muted
                    if synth:
                        synth.muted = muted
                elif key == 'p':
                    # toggle pause
                    is_paused[0] = not is_paused[0]
                    lcd.paused = is_paused[0]
                    if is_paused[0]:
                        # pause: stop FM reader, silence synth
                        fm.stop()
                        if synth:
                            synth.set_pitch(0)
                        print("\r  ** PAUSED **                                          ")
                    else:
                        # unpause: start FM reader
                        fm.start()
                        print(f"\r  ** RESUMED — FM {fm.freq} **                        ")
                elif key == 'f':
                    # cycle FM preset
                    current_preset[0] = (current_preset[0] + 1) % len(FM_PRESETS)
                    new_freq = FM_PRESETS[current_preset[0]]
                    lcd.fm_freq = new_freq
                    if not is_paused[0]:
                        fm.set_freq(new_freq)
                    else:
                        fm.freq = new_freq
                    print(f"\r  ** FM -> {new_freq} **                                  ")
                elif key == 'x':
                    # reset conf and gate to defaults (joystick hold)
                    pipeline.conf_th = args.conf
                    pipeline.rms_th = args.rms
                    print(f"\r  ** RESET — conf {args.conf:.1f}  gate {args.rms:.3f} **       ")
                elif key == 'Q':
                    # quit (hold KEY1 3s, only when paused)
                    if is_paused[0]:
                        print("\r  ** SHUTTING DOWN **                                   ")
                        break

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

                # update LCD display state
                lcd.conf_th = pipeline.conf_th
                lcd.rms_th = pipeline.rms_th
                lcd.note_name = midi_to_name(vf.current_midi) if vf.current_midi > 0 else "--"
                lcd.freq_hz = vf.current_f0
                lcd.note_count = note_count
                lcd.mode = "RADIO" if radio_mode[0] else "GHOST"
                lcd.muted = muted
                lcd.push_note(vf.current_midi)

                line = (
                    f"\r  FM {fm.freq:>7s}"
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
        joy.stop()
        lcd.stop()
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
