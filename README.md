# The Digital Ear

A streaming audio-to-MIDI extraction pipeline built for [Paradromics](https://paradromics.com) Qualifier. Turns a raw audio signal into discrete MIDI note events using harmonic analysis, source separation, and an online Viterbi decoder — all running in constant memory on a single thread.

Built to eventually run on a Raspberry Pi in real time.

> **[Watch the walkthrough (YouTube)](youtube.com)** · **[Slide deck (Google Slides)](https://docs.google.com/presentation/d/1e4HgQPgvp3ZzGGHwky6CwUFruW1vtq-2nBTBd-b3gio/edit?usp=sharing)**

![GUI Screenshot](other/gui_screenshot.png)

---

## How it works

```
Audio in → Preprocess → HPSS → Pitch Detection → Viterbi → Note Tracking → MIDI out
```

Each stage is streaming. Each stage has bounded memory. Audio goes in one end as 2048-sample blocks, MIDI notes come out the other.

### The pipeline

**Preprocessing** — DC blocker, high-pass at 60 Hz, low-pass at 4 kHz. Three cascaded 1-pole IIR filters. Removes rumble and high-frequency harmonics that confuse pitch detection.

**HPSS (Harmonic-Percussive Source Separation)** — Median filtering on a sliding STFT buffer. Separates the tonal content (voice, guitar, piano) from transients (drums, clicks, plucks). The pitch detector only sees the harmonic stream. Based on [Fitzgerald, DAFx 2010].

**Pitch Detection** — MELODIA-style harmonic salience function. Builds a log-frequency salience map (10-cent resolution), sums energy across harmonics with cos² spreading, applies A-weighting, and extracts the top candidates with confidence scores. Derived from [Salamon & Gomez, IEEE 2012].

**Online Viterbi** — Fixed-lag HMM decoder that picks the best pitch candidate at each frame while enforcing temporal continuity. The transition and emission costs adapt in real time based on two causal estimates:
- *Tonality* (0.5s lookback) — how tonal vs. noisy the signal is right now
- *Voicing density* (10s lookback) — what fraction of recent frames had pitch candidates

This density-adaptive behavior is custom — not from any paper. It lets the decoder tighten up during clean melodic passages and loosen during noisy sections, without any manual threshold tuning.

**Note Tracking** — Median smoothing, f0-to-MIDI quantization, run-length encoding, minimum duration gating (120 ms), fragment merging, and octave correction. Converts the raw frame-by-frame pitch stream into clean note-on/note-off events.

### Pipeline numbers

| Metric | Value |
|---|---|
| Total latency | ~753 ms (audio in → MIDI decision) |
| Memory (constant) | ~31 MB regardless of audio length |
| Real-time factor | 10-20x faster than real-time on laptop |
| Block size | 2048 samples (46 ms at 44.1 kHz) |
| Viterbi lag | 50 frames (~580 ms) |

---

## Quick start

### Requirements

- Python 3.7+
- NumPy
- ffmpeg (on your PATH)
- tkinter (included with Python on most systems)

```bash
pip install numpy psutil
```

That's it. No scipy, no librosa, no tensorflow.

### Run the GUI

```bash
python gui.py
```

Pick an input file, hit Generate. The output MIDI lands in `outputs/`.

### Run from the command line

```bash
# Basic — extract melody to MIDI
python main.py --in "song.m4a" --out output.mid

# With debug stats
python main.py --in "song.m4a" --out output.mid --debug

# Export a side-by-side WAV (original left, synth right)
python main.py --in "song.m4a" --out output.mid --dual output_dual.wav

# Polyphonic mode (extracts up to 3 voices)
python main.py --in "song.m4a" --out output.mid --poly

# GUI usage to easily exapnded functions
python gui.py
```

### CLI options

| Flag | Default | What it does |
|---|---|---|
| `--in` | *(required)* | Input audio file (.m4a, .wav, .mp3, .flac, .ogg) |
| `--out` | *(required)* | Output MIDI file path |
| `--sr` | 44100 | Sample rate |
| `--fmin` | 80 | Lowest pitch to detect (Hz) |
| `--fmax` | 1000 | Highest pitch to detect (Hz) |
| `--conf-th` | 7.0 | Pitch confidence threshold |
| `--debug` | off | Print timing, memory, voicing stats |
| `--poly` | off | Polyphonic extraction (3 voices) |
| `--wav` | — | Also export a synthesized WAV |
| `--dual` | — | Stereo WAV: original + synth side by side |
| `--dump-frames` | — | CSV with per-frame pitch/confidence/RMS |

---

## Project structure

```
the-digital-ear/
├── main.py                     # CLI entry point
├── gui.py                      # Tkinter GUI
├── generate_spectrogram.py     # Spectrogram visualization tool
│
├── digital_ear/
│   ├── audio_io.py             # ffmpeg streaming decoder
│   ├── preprocess.py           # DC block, HPF, LPF (1-pole IIR)
│   ├── hpss.py                 # Harmonic-Percussive Source Separation
│   ├── features.py             # Frame extraction
│   ├── harmonic_pitch.py       # MELODIA-style pitch salience
│   ├── melody_extractor.py     # Online Viterbi decoder
│   ├── note_tracker.py         # f0 stream → MIDI note events
│   ├── midi_writer.py          # Raw binary MIDI writer (no deps)
│   └── perf.py                 # Performance/memory profiler
│
├── tests/
│   ├── test_hps_pitch.py
│   ├── test_hps_unvoiced.py
│   └── test_tone_fft.py
│
└── outputs/                    # Generated MIDI, WAV, spectrograms
```

---

## Spectrogram walkthrough

Generated with `generate_spectrogram.py`, these show what happens at each pipeline stage on the same audio clip:

| Stage | What you're seeing |
|---|---|
| **1. Raw audio** | Unprocessed FFT — everything mixed together |
| **2. Preprocessed** | After DC block + HPF + LPF — rumble and highs removed |
| **3. After HPSS** | Harmonic stream only — percussion stripped out |
| **4. Pitch overlay** | Extracted pitch contour (cyan) on the harmonic spectrogram |
| **5. MIDI piano roll** | Final quantized note events |

---

## Papers this builds on

| Problem | Paper | What we took |
|---|---|---|
| Percussion bleeds into pitch | Fitzgerald, DAFx 2010 | HPSS via median filtering |
| Pitch ambiguity / harmonics | Salamon & Gomez, IEEE 2012 | Harmonic salience function |
| Frame-to-frame pitch flicker | Mauch & Dixon, ICASSP 2014 | HMM + Viterbi smoothing |
| Static parameters fail on mixed audio | *(custom)* | Density-adaptive transition/emission costs |

---

## What's next (Stage 2)

The DSP pipeline is fully streaming and constant-memory. What's left is the I/O layer:

- **Live audio input** — Replace file decoding with microphone / hardware stream
- **Real-time MIDI output** — Emit note events to a MIDI port as they close (~860 ms latency)
- **NoteTracker bounded memory** — Convert growing lists to bounded deques (same pattern as every other stage)
- **RPi benchmarking** — Verify the pipeline runs faster than real-time on target hardware

---

## License

Internal project for Paradromics. Not currently open-sourced.
