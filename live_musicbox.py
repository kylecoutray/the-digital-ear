#!/usr/bin/env python3
"""
The Digital Ear — Live Musicbox
Real-time audio-to-MIDI with live visualization and synthesis.

Records from the microphone, runs the streaming pipeline, plays back
detected notes as sine tones, and shows a scrolling piano roll.

  python live_musicbox.py
  python live_musicbox.py --device 2        # specific input device
  python live_musicbox.py --no-synth        # disable audio playback
  python live_musicbox.py --list-devices    # show available audio devices
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
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


# --- constants & palette (matches gui.py) ---

SR = 44100
BLOCK_SIZE = 2048
N_FFT = 2048
HOP = 512

BG         = "#0a0a0a"
BG_CARD    = "#141416"
FG         = "#e8e0d4"
FG_DIM     = "#7a7a82"
FG_MUTED   = "#9a9aa0"
ACCENT     = "#c8402d"
BORDER     = "#2a2a2e"
NEON_GREEN = "#39ff14"
NEON_DIM   = "#1a8a0a"
KEY_WHITE  = "#3a3a42"
KEY_BLACK  = "#222228"
PLAYHEAD   = "#c8402d"
NOTE_GLOW  = "#39ff14"
METER_LO   = "#39ff14"
METER_MID  = "#e8d44d"
METER_HI   = "#c8402d"
STATUS_BG  = "#111114"

AURA = [
    (240, 140, 30), (210, 100, 20), (80, 50, 160),
    (30, 70, 210), (50, 100, 240), (70, 80, 220),
    (140, 50, 180), (210, 55, 65), (180, 40, 35),
]
GLOW_PX = 10

MIDI_LO = 48   # C3
MIDI_HI = 84   # C6
NUM_KEYS = MIDI_HI - MIDI_LO

ROLL_SECONDS = 8.0  # seconds of history to show

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def is_black_key(m: int) -> bool:
    return (m % 12) in {1, 3, 6, 8, 10}


def lerp_color(c1: tuple, c2: tuple, t: float) -> str:
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# viz data, shared between pipeline thread and gui

@dataclass
class VizFrame:
    """One frame of pipeline state for the GUI."""
    time_sec: float = 0.0
    audio_time: float = 0.0      # same base as NoteEvent timestamps
    rms_level: float = 0.0
    current_f0: float = 0.0      # 0 = unvoiced
    current_midi: int = 0
    confidence: float = 0.0
    notes: list = field(default_factory=list)


# --- Sine Synth ---

class SineSynth:
    """Thread-safe sine synth with pitch-following and a 2s safety timeout.

    set_pitch(f0_hz) for continuous tracking, play_note(midi, dur) for one-shots.
    muted flag is checked in generate() for instant mute from GUI thread.
    """

    NOTE_MAX_SEC = 3.0  # hard cap — no note plays longer than this

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

        # smoothing coeffs
        self._attack_coeff = 1.0 - math.exp(-2.0 * math.pi * 30.0 / sr)
        self._release_coeff = 1.0 - math.exp(-2.0 * math.pi * 20.0 / sr)

    def set_pitch(self, f0_hz: float):
        """Call each frame. f0 > 20 plays + resets 2s timeout, else fades out."""
        with self._lock:
            if f0_hz > 20:
                self._freq = f0_hz
                self._target_amp = 0.30
                # reset timeout
                self._note_end_sample = self._sample_counter + int(2.0 * self.sr)
            else:
                self._target_amp = 0.0

    def play_note(self, midi_note: int, duration_sec: float):
        """One-shot note, auto-stops after duration_sec (capped)."""
        duration_sec = min(duration_sec, self.NOTE_MAX_SEC)
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        with self._lock:
            self._freq = freq
            self._target_amp = 0.30
            self._note_end_sample = self._sample_counter + int(duration_sec * self.sr)

    def stop(self):
        """Kill it."""
        with self._lock:
            self._target_amp = 0.0

    def generate(self, frames: int) -> np.ndarray:
        """Generate `frames` samples. Called from output callback."""
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
            # note expired, kill it
            if sc + i >= note_end:
                target_amp = 0.0

            # smooth amp
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


# Pipeline Thread

class PipelineRunner:
    """Runs the full Digital Ear pipeline in a background thread."""

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

        # gui-tunable
        self.conf_th: float = conf_th
        self.rms_th: float = rms_th
        self.synth_muted: bool = False

        # pipeline setup (mirrors main.py)
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

            if blk is None:  # Poison pill
                break

            new_notes: list[NoteEvent] = []
            wall_sec = time.monotonic() - self._start_time

            # 1: preprocess
            y = self.pre.process(blk)

            # kill synth if very quiet
            self._latest_rms = rms(y)
            if self._latest_rms < self.rms_th * 0.5 and self.synth and not self.synth_muted:
                self.synth.set_pitch(0)

            # 2: HPSS
            self.hpss.push(y)

            # 3-6: pitch detection -> viterbi -> note tracker
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
                        # min_conf=1.5 floor for candidates (matches main.py)
                        candidates, tonality = self.det.estimate_candidates(
                            frame, n_top=5, min_conf=1.5
                        )
                        # confidence gate from slider (1.0 = no filtering)
                        if conf_th > 1.5:
                            candidates = [(hz, c) for hz, c in candidates
                                          if c >= conf_th]
                        self._silent_frames = 0

                    f0_hz = self.melody.push(candidates, tonality)

                    # kill synth during extended silence (don't override f0, let viterbi decide)
                    if self._silent_frames > 8:
                        if self.synth and not self.synth_muted:
                            self.synth.set_pitch(0)

                    if f0_hz is not None:
                        # drive synth with raw f0 (lowest latency)
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

                        # pass None not 0.0 for unvoiced — 0.0 poisons median buffer (log2(0) crash)
                        tracker_f0 = f0_hz if f0_hz > 0 else None
                        events = self.tracker.push(tracker_f0)
                        new_notes.extend(events)

                    elif not candidates:
                        # nothing at all — clear key display instead of holding stale note
                        self._latest_midi = 0
                        self._latest_f0 = 0.0
                        self._latest_conf = 0.0
                        if self.synth and not self.synth_muted:
                            self.synth.set_pitch(0)

                self._stream_sample_idx += len(harmonic_block)

            # push viz frame (audio_time is same clock as NoteEvent timestamps)
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
                pass  # gui behind, drop

    def stop(self):
        self.running = False


# ---- Live Musicbox (tk app) ----

class LiveMusicbox(tk.Tk):
    """Real-time audio-to-MIDI visualizer and synth."""

    def __init__(self, device: Optional[int] = None, enable_synth: bool = True):
        super().__init__()

        self.title("The Digital Ear — Live Musicbox")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(900, 500)
        self.geometry("1400x800")

        # fonts
        self._font_title = tkfont.Font(family="Helvetica Neue", size=22, weight="bold")
        self._font_note = tkfont.Font(family="Menlo", size=28, weight="bold")
        self._font_label = tkfont.Font(family="Helvetica Neue", size=15)
        self._font_key = tkfont.Font(family="Menlo", size=8)
        self._font_stat = tkfont.Font(family="Menlo", size=13)
        self._font_brand = tkfont.Font(family="PT Mono", size=18)

        # state
        self._all_notes: list[NoteEvent] = []
        self._current_midi = 0
        self._current_rms = 0.0
        self._current_f0 = 0.0
        self._wall_time = 0.0
        self._audio_time = 0.0           # same clock as NoteEvent times
        self._audio_time_wall = 0.0      # wall-clock when _audio_time last updated
        self._note_count = 0
        self._running = False
        self._aura_offset = 0.0
        self._glow_last_t = 0.0
        self._keys_last_midi = -1        # skip redraw if unchanged

        # live note (extends in real-time before tracker finalizes)
        self._live_note_midi: int = 0       # 0 = none
        self._live_note_start: float = 0.0

        # queues
        self._audio_q: queue.Queue = queue.Queue(maxsize=64)
        self._viz_q: queue.Queue = queue.Queue(maxsize=128)

        # synth
        self._synth = SineSynth() if enable_synth else None
        self._enable_synth = enable_synth

        # audio
        self._device = device

        # ui
        self._build_ui()
        self._in_stream: Optional[sd.InputStream] = None
        self._out_stream: Optional[sd.OutputStream] = None

        # pipeline
        self._pipeline = PipelineRunner(
            self._audio_q, self._viz_q, synth=self._synth,
        )
        self._pipeline_thread: Optional[threading.Thread] = None

        # kick it off
        self.after(100, self._start_streams)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- ui layout ---

    def _build_ui(self):
        # outer frame w/ glow border
        self._outer = tk.Frame(self, bg=BG)
        self._outer.pack(fill="both", expand=True)

        # glow canvas
        self._glow_canvas = tk.Canvas(
            self._outer, bg=BG, highlightthickness=0, bd=0
        )
        self._glow_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        # main container
        self._main = tk.Frame(self._outer, bg=BG_CARD)
        self._main.place(
            x=GLOW_PX, y=GLOW_PX,
            relwidth=1, relheight=1,
            width=-2 * GLOW_PX, height=-2 * GLOW_PX,
        )

        title_frame = tk.Frame(self._main, bg=BG_CARD, height=70)
        title_frame.pack(fill="x", side="top")
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame, text="The Digital Ear", font=self._font_title,
            fg=FG, bg=BG_CARD, anchor="w", padx=16,
        ).pack(side="left", fill="y")

        tk.Label(
            title_frame, text="LIVE MUSICBOX", font=self._font_brand,
            fg=ACCENT, bg=BG_CARD, anchor="e", padx=16,
        ).pack(side="right", fill="y")

        tk.Frame(self._main, bg=BORDER, height=1).pack(fill="x")

        # control strip
        ctrl_frame = tk.Frame(self._main, bg=BG_CARD, height=60)
        ctrl_frame.pack(fill="x", side="top")
        ctrl_frame.pack_propagate(False)

        self._font_ctrl = tkfont.Font(family="Helvetica Neue", size=20)

        tk.Label(
            ctrl_frame, text="Confidence", font=self._font_ctrl,
            fg=FG_DIM, bg=BG_CARD,
        ).pack(side="left", padx=(16, 4), pady=4)

        self._conf_var = tk.DoubleVar(value=1.0)
        conf_scale = tk.Scale(
            ctrl_frame, from_=1.0, to=25.0, resolution=0.5,
            orient="horizontal", variable=self._conf_var,
            bg=BG_CARD, fg=FG, troughcolor="#1a1a1e",
            highlightthickness=0, bd=0, width=10, length=140,
            showvalue=True, font=self._font_ctrl,
            command=self._on_conf_change,
        )
        conf_scale.pack(side="left", padx=(0, 12), pady=2)

        tk.Label(
            ctrl_frame, text="(↑ = pickier)", font=self._font_ctrl,
            fg=FG_DIM, bg=BG_CARD,
        ).pack(side="left", padx=(0, 24))

        tk.Label(
            ctrl_frame, text="Noise Gate", font=self._font_ctrl,
            fg=FG_DIM, bg=BG_CARD,
        ).pack(side="left", padx=(0, 4), pady=4)

        self._rms_var = tk.DoubleVar(value=0.003)
        rms_scale = tk.Scale(
            ctrl_frame, from_=0.001, to=0.1, resolution=0.001,
            orient="horizontal", variable=self._rms_var,
            bg=BG_CARD, fg=FG, troughcolor="#1a1a1e",
            highlightthickness=0, bd=0, width=10, length=140,
            showvalue=True, font=self._font_ctrl,
            command=self._on_rms_change,
        )
        rms_scale.pack(side="left", padx=(0, 12), pady=2)

        tk.Label(
            ctrl_frame, text="(↑ = quieter ignored)", font=self._font_ctrl,
            fg=FG_DIM, bg=BG_CARD,
        ).pack(side="left", padx=(0, 24))

        self._synth_muted_var = tk.BooleanVar(value=False)
        self._mute_btn = tk.Checkbutton(
            ctrl_frame, text="Mute Synth", font=self._font_ctrl,
            variable=self._synth_muted_var,
            fg=FG, bg=BG_CARD, selectcolor=BG,
            activebackground=BG_CARD, activeforeground=FG,
            highlightthickness=0, bd=0,
            command=self._on_mute_toggle,
        )
        self._mute_btn.pack(side="right", padx=16)

        tk.Frame(self._main, bg=BORDER, height=1).pack(fill="x")

        # keys + piano roll
        middle = tk.Frame(self._main, bg=BG_CARD)
        middle.pack(fill="both", expand=True)

        self._key_canvas = tk.Canvas(
            middle, bg=BG_CARD, width=50, highlightthickness=0, bd=0
        )
        self._key_canvas.pack(side="left", fill="y")

        self._roll_canvas = tk.Canvas(
            middle, bg=BG, highlightthickness=0, bd=0
        )
        self._roll_canvas.pack(side="left", fill="both", expand=True)

        tk.Frame(self._main, bg=BORDER, height=1).pack(fill="x")

        # status bar
        status_frame = tk.Frame(self._main, bg=STATUS_BG, height=60)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)

        self._note_label = tk.Label(
            status_frame, text="--", font=self._font_note,
            fg=NEON_GREEN, bg=STATUS_BG, width=5, anchor="center",
        )
        self._note_label.pack(side="left", padx=(16, 8))

        self._meter_canvas = tk.Canvas(
            status_frame, bg=STATUS_BG, width=180, height=20,
            highlightthickness=0, bd=0
        )
        self._meter_canvas.pack(side="left", padx=8, pady=18)

        self._stat_label = tk.Label(
            status_frame, text="", font=self._font_stat,
            fg=FG_DIM, bg=STATUS_BG, anchor="w",
        )
        self._stat_label.pack(side="left", padx=16, fill="x", expand=True)

        self._listening_label = tk.Label(
            status_frame, text="", font=self._font_label,
            fg=NEON_GREEN, bg=STATUS_BG, anchor="e", padx=16,
        )
        self._listening_label.pack(side="right")

        # device selector
        self._device_var = tk.StringVar()
        self._device_map: dict[str, int | None] = {}  # name -> device idx
        self._populate_device_list()

        self._device_menu = tk.OptionMenu(
            status_frame, self._device_var,
            *self._device_map.keys(),
            command=self._on_device_change,
        )
        self._device_menu.configure(
            font=self._font_key, bg=STATUS_BG, fg=FG,
            activebackground="#2a2a2e", activeforeground=FG,
            highlightthickness=0, relief="flat", bd=0,
            width=30, anchor="e",
        )
        self._device_menu["menu"].configure(
            bg="#1a1a1e", fg=FG, activebackground=ACCENT,
            activeforeground=FG, font=self._font_key,
        )
        self._device_menu.pack(side="right", padx=(0, 4))

    def _populate_device_list(self):
        """Build device name -> index map."""
        devices = sd.query_devices()
        self._device_map = {}

        # default input device
        try:
            default_idx = sd.default.device[0]  # input device index
        except Exception:
            default_idx = None

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                name = f"{dev['name']}"
                # dedup
                if name in self._device_map:
                    name = f"{name} ({i})"
                self._device_map[name] = i

        if self._device is not None:
            for name, idx in self._device_map.items():
                if idx == self._device:
                    self._device_var.set(name)
                    break
        elif default_idx is not None:
            for name, idx in self._device_map.items():
                if idx == default_idx:
                    self._device_var.set(name)
                    break

    def _on_device_change(self, selected_name):
        new_idx = self._device_map.get(selected_name)
        if new_idx == self._device:
            return

        self._device = new_idx
        self._listening_label.config(text="SWITCHING...", fg=METER_MID)

        # tear down old stream
        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass

        # open new one
        try:
            self._in_stream = sd.InputStream(
                samplerate=SR,
                channels=1,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                device=self._device,
                callback=self._audio_input_callback,
            )
            self._in_stream.start()
            self._listening_label.config(text="LISTENING", fg=NEON_GREEN)
        except Exception as e:
            self._listening_label.config(text=f"Mic error: {e}", fg=ACCENT)

    # slider callbacks

    def _on_conf_change(self, val):
        self._pipeline.conf_th = float(val)

    def _on_rms_change(self, val):
        self._pipeline.rms_th = float(val)

    def _on_mute_toggle(self):
        """Goes straight to synth, bypasses pipeline."""
        muted = self._synth_muted_var.get()
        self._pipeline.synth_muted = muted
        if self._synth:
            self._synth.muted = muted

    # audio streams

    def _start_streams(self):
        try:
            self._in_stream = sd.InputStream(
                samplerate=SR,
                channels=1,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                device=self._device,
                callback=self._audio_input_callback,
            )
            self._in_stream.start()
        except Exception as e:
            self._listening_label.config(text=f"Mic error: {e}", fg=ACCENT)
            return

        if self._enable_synth and self._synth:
            try:
                self._out_stream = sd.OutputStream(
                    samplerate=SR,
                    channels=1,
                    blocksize=1024,
                    dtype="float32",
                    callback=self._audio_output_callback,
                )
                self._out_stream.start()
            except Exception:
                pass  # synth is optional

        # pipeline thread
        self._pipeline_thread = threading.Thread(
            target=self._pipeline.run, daemon=True
        )
        self._pipeline_thread.start()

        self._running = True
        self._listening_label.config(text="LISTENING", fg=NEON_GREEN)

        # go
        self._update_loop()

    def _audio_input_callback(self, indata, frames, time_info, status):
        """sd input callback (audio thread)."""
        if status:
            pass
        try:
            self._audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # pipeline behind, drop

    def _audio_output_callback(self, outdata, frames, time_info, status):
        """sd output callback (audio thread)."""
        if self._synth:
            outdata[:, 0] = self._synth.generate(frames)
        else:
            outdata.fill(0)

    # --- gui update loop ---

    def _update_loop(self):
        """Poll pipeline and redraw (~16ms)."""
        if not self._running:
            return

        # drain viz queue
        updated = False
        while True:
            try:
                vf: VizFrame = self._viz_q.get_nowait()
                self._wall_time = vf.time_sec
                self._current_rms = vf.rms_level
                self._current_f0 = vf.current_f0
                prev_midi = self._current_midi
                self._current_midi = vf.current_midi

                # audio time (same clock as NoteEvent)
                self._audio_time = vf.audio_time
                self._audio_time_wall = time.monotonic()

                # live note tracking
                if vf.current_midi != self._live_note_midi:
                    # pitch changed, finalize previous
                    if self._live_note_midi > 0:
                        if vf.audio_time - self._live_note_start > 0.05:
                            self._all_notes.append(NoteEvent(
                                note=self._live_note_midi,
                                start_sec=self._live_note_start,
                                end_sec=vf.audio_time,
                            ))
                        self._live_note_midi = 0

                    # start new one if voiced
                    if vf.current_midi > 0:
                        self._live_note_midi = vf.current_midi
                        self._live_note_start = vf.audio_time

                for note in vf.notes:
                    self._all_notes.append(note)
                    self._note_count += 1

                updated = True
            except queue.Empty:
                break

        now_wall = time.monotonic()

        # glow: throttle to ~20fps, doesn't need 60
        if now_wall - self._glow_last_t >= 0.05:
            self._draw_glow()
            self._glow_last_t = now_wall

        # keys: only redraw on note change
        if self._current_midi != self._keys_last_midi:
            self._draw_keys()
            self._keys_last_midi = self._current_midi

        self._draw_piano_roll()
        self._draw_meter()
        self._update_status()

        self.after(16, self._update_loop)

    # glow border

    def _draw_glow(self):
        c = self._glow_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        c.delete("glow")
        self._aura_offset += 0.003

        n_seg = len(AURA) - 1
        perimeter = 2 * (w + h)

        for edge_px in range(0, perimeter, 8):
            t = ((edge_px / perimeter) + self._aura_offset) % 1.0
            seg = t * n_seg
            idx = int(seg)
            frac = seg - idx
            idx = min(idx, n_seg - 1)
            color = lerp_color(AURA[idx], AURA[idx + 1], frac)

            if edge_px < w:  # top
                x, y = edge_px, 0
                x2, y2 = min(edge_px + 9, w), GLOW_PX
            elif edge_px < w + h:  # right
                x, y = w - GLOW_PX, edge_px - w
                x2, y2 = w, min(edge_px - w + 9, h)
            elif edge_px < 2 * w + h:  # bottom
                x, y = w - (edge_px - w - h), h - GLOW_PX
                x2, y2 = min(x + 9, w), h
            else:  # left
                x, y = 0, h - (edge_px - 2 * w - h)
                x2, y2 = GLOW_PX, min(y + 9, h)

            c.create_rectangle(x, y, x2, y2, fill=color, outline="", tags="glow")

    # piano keys

    def _draw_keys(self):
        c = self._key_canvas
        h = c.winfo_height()
        w = c.winfo_width()
        if h < 10:
            return

        c.delete("all")

        key_h = h / NUM_KEYS
        for i in range(NUM_KEYS):
            midi = MIDI_HI - 1 - i
            y = i * key_h
            black = is_black_key(midi)
            is_active = (midi == self._current_midi and self._current_midi > 0)

            if is_active:
                fill = NEON_GREEN
                text_fill = BG
            elif black:
                fill = KEY_BLACK
                text_fill = "#5a5a64"
            else:
                fill = KEY_WHITE
                text_fill = FG

            c.create_rectangle(0, y, w, y + key_h, fill=fill, outline=BORDER, width=0.5)

            # label white keys + active note
            if not black or is_active:
                name = midi_to_name(midi)
                c.create_text(
                    w - 4, y + key_h / 2, text=name,
                    fill=text_fill, font=self._font_key, anchor="e"
                )

    # --- piano roll ---

    def _draw_piano_roll(self):
        c = self._roll_canvas
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw < 10 or ch < 10:
            return

        c.delete("all")

        key_h = ch / NUM_KEYS
        # smooth interpolation between pipeline updates for jitter-free scroll
        elapsed = time.monotonic() - self._audio_time_wall if self._audio_time_wall > 0 else 0.0
        now = self._audio_time + elapsed
        px_per_sec = cw / ROLL_SECONDS

        # playhead
        playhead_x = cw - 40

        # grid lines (octave C's)
        for i in range(NUM_KEYS):
            midi = MIDI_HI - 1 - i
            y = i * key_h
            if midi % 12 == 0:
                c.create_line(0, y, cw, y, fill=BORDER, width=0.5)

        for note_evt in self._all_notes:
            midi = note_evt.note
            if midi < MIDI_LO or midi >= MIDI_HI:
                continue

            # time -> x
            note_start_x = playhead_x - (now - note_evt.start_sec) * px_per_sec
            note_end_x = playhead_x - (now - note_evt.end_sec) * px_per_sec

            # off-screen
            if note_end_x < -10 or note_start_x > cw + 10:
                continue

            row = MIDI_HI - 1 - midi
            y = row * key_h + 1
            h = key_h - 2

            # fade with age
            age = now - note_evt.end_sec
            if age < 0.5:
                fill = NEON_GREEN
                outline_c = "#4dff30"
            elif age < 2.0:
                t = (age - 0.5) / 1.5
                fill = lerp_color((57, 255, 20), (40, 120, 20), t)
                outline_c = ""
            else:
                t = min((age - 2.0) / 3.0, 1.0)
                fill = lerp_color((40, 120, 20), (42, 42, 48), t)
                outline_c = ""

            c.create_rectangle(
                note_start_x, y, note_end_x, y + h,
                fill=fill, outline=outline_c, width=1 if outline_c else 0,
            )

        # live note extends from onset to playhead
        if self._live_note_midi > 0 and MIDI_LO <= self._live_note_midi < MIDI_HI:
            row = MIDI_HI - 1 - self._live_note_midi
            y = row * key_h + 1
            h = key_h - 2
            live_start_x = playhead_x - (now - self._live_note_start) * px_per_sec
            live_start_x = max(0, live_start_x)
            c.create_rectangle(
                live_start_x, y, playhead_x, y + h,
                fill=NEON_GREEN, outline="#4dff30", width=1,
            )

        # playhead line
        c.create_line(
            playhead_x, 0, playhead_x, ch,
            fill=PLAYHEAD, width=2, dash=(6, 3)
        )

        # now label
        c.create_text(
            playhead_x, ch - 6, text="NOW",
            fill=ACCENT, font=self._font_key, anchor="s"
        )

        # trim old notes
        cutoff = now - ROLL_SECONDS * 2.5
        if self._all_notes and self._all_notes[0].end_sec < cutoff:
            trim = 0
            for n in self._all_notes:
                if n.end_sec < cutoff:
                    trim += 1
                else:
                    break
            self._all_notes = self._all_notes[trim:]

    # level meter

    def _draw_meter(self):
        c = self._meter_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10:
            return

        c.delete("all")

        c.create_rectangle(0, 0, w, h, fill="#1a1a1e", outline=BORDER)

        # rms -> bar width (log scale)
        level = self._current_rms
        if level > 0.001:
            db = 20 * math.log10(level + 1e-10)
            bar = max(0, min(1.0, (db + 60) / 60))  # -60dB → 0dB maps to 0→1
        else:
            bar = 0.0

        bar_w = int(bar * (w - 4))

        # color
        if bar < 0.6:
            color = METER_LO
        elif bar < 0.85:
            color = METER_MID
        else:
            color = METER_HI

        if bar_w > 0:
            c.create_rectangle(2, 2, 2 + bar_w, h - 2, fill=color, outline="")

    def _update_status(self):
        if self._current_midi > 0:
            name = midi_to_name(self._current_midi)
            self._note_label.config(text=name, fg=NEON_GREEN)
        else:
            self._note_label.config(text="--", fg=FG_DIM)

        rms_str = f"RMS {self._current_rms:.4f}"
        f0_str = f"f0 {self._current_f0:.1f} Hz" if self._current_f0 > 0 else "f0 --"
        notes_str = f"{self._note_count} notes"
        time_str = f"{self._wall_time:.1f}s"
        synth_str = "SYNTH ON" if self._enable_synth else "SYNTH OFF"

        self._stat_label.config(
            text=f"  {f0_str}  |  {rms_str}  |  {notes_str}  |  {time_str}  |  {synth_str}"
        )

    def _on_close(self):
        self._running = False

        # pipeline
        self._pipeline.stop()
        try:
            self._audio_q.put_nowait(None)  # Poison pill
        except queue.Full:
            pass

        # streams
        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass

        if self._out_stream:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass

        # wait for thread
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=1.0)

        self.destroy()


# main

def main():
    parser = argparse.ArgumentParser(
        description="The Digital Ear — Live Musicbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Input audio device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--no-synth", action="store_true",
        help="Disable audio playback (visualization only)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    app = LiveMusicbox(device=args.device, enable_synth=not args.no_synth)
    app.mainloop()


if __name__ == "__main__":
    main()
