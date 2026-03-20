# Stage 2: Ghost FM Synthesizer — Hardware Reference

## What This Doc Is
Everything needed to buy exact, compatible parts for Stage 2.
Drop this into any chat, hand it to a store clerk, whatever.

---

## PROJECT SUMMARY

Take the existing Digital Ear pipeline (Python, real-time audio → pitch detection → synth) and run it on a Raspberry Pi fed by FM radio instead of a microphone. The device listens to FM, extracts the melody, and re-synthesizes it with a ghostly timbre. Standalone, headless, physical controls.

---

## EXISTING SOFTWARE STACK

- **Python 3.10** (ARM64 compatible — currently running on Apple Silicon)
- **Core DSP dependencies**: numpy, scipy (heavy lifting — FFT, filtering)
- **Audio I/O**: sounddevice (wraps PortAudio)
- **Other**: mido (MIDI file writing, optional for Stage 2), psutil (perf monitoring)
- **NO librosa in the real-time path** — all DSP is custom numpy
- **GUI**: tkinter (will be stripped for headless, replaced with GPIO controls)
- **Synth**: Custom SineSynth class using sounddevice output streams

### Pipeline (streaming, per-frame):
```
Audio In (2048 samples @ 44100 Hz, hop 512)
  → Preprocessor (normalize, DC removal)
  → HPSS (harmonic/percussive separation)
  → FrameExtractor (spectral features)
  → HarmonicPitchDetector.estimate_candidates(n_top=5, min_conf=1.5)
  → MelodyExtractor (Viterbi smoothing)
  → NoteTracker (min_note_sec=0.15)
  → SineSynth output
```

### Key Parameters:
- Sample rate: 44100 Hz
- Block size: 2048 samples
- FFT size: 2048
- Hop size: 512
- RMS silence threshold: 0.003

---

## BILL OF MATERIALS — EXACT PARTS

### 1. COMPUTE — Raspberry Pi 5 (4GB)

| Part | Exact Product | Price (USD) | Notes |
|------|--------------|-------------|-------|
| Pi 5 board | **Raspberry Pi 5, 4GB RAM** | ~$60 | 2.4GHz quad-core Cortex-A76. The 4GB model is plenty — DSP is CPU-bound, not memory-bound. Do NOT get Pi 4 — the A76 cores are ~2-3x faster than Pi 4's A72 for numpy workloads. |
| Power supply | **Raspberry Pi 27W USB-C Power Supply** (official) | ~$12 | MUST be 5V/5A (27W). Third-party 5V/3A supplies will throttle under SDR + DSP load. Get the official one. |
| MicroSD | **SanDisk Extreme 32GB microSDHC** (A2 rated) | ~$10 | A2 rating matters for random I/O during Python imports. 32GB is more than enough. |
| Heatsink/cooling | **Raspberry Pi 5 Active Cooler** (official) | ~$5 | The official clip-on fan + heatsink. Sustained DSP will thermal throttle without active cooling. Non-negotiable. |
| Case | **Raspberry Pi 5 Case** (official, fits active cooler) | ~$10 | Or any case that fits the active cooler. The official one has a built-in fan duct. |

**Subtotal: ~$97**

### 2. FM RECEPTION — RTL-SDR

| Part | Exact Product | Price (USD) | Notes |
|------|--------------|-------------|-------|
| SDR dongle | **RTL-SDR Blog V4** (R828D tuner + RTL2832U) | ~$30 | The V4 specifically — has HF direct sampling, better filtering, and a TCXO for frequency stability. Comes with SMA connector. Covers 24-1766 MHz (FM broadcast 88-108 MHz is well within range). |
| Antenna | **RTL-SDR Blog Dipole Antenna Kit** | ~$12 | Comes with the V4 kit usually. If buying separately, get telescoping dipole with SMA base. Set each element to ~75cm for FM band (~97 MHz quarter-wave). |
| SMA adapter | Included with RTL-SDR Blog V4 | $0 | V4 uses SMA female. Antenna kit matches. No adapter needed if you buy both from RTL-SDR Blog. |

**Subtotal: ~$42**

### FM → Audio Integration (software, not hardware):
```bash
# rtl_fm demodulates FM to raw PCM audio on stdout
# install: sudo apt install rtl-sdr
rtl_fm -f 99.5M -M fm -s 170k -r 44100 -A fast -l 0 -
# outputs: signed 16-bit LE mono PCM @ 44100 Hz to stdout
# pipe directly into Python script via subprocess or stdin
```

### 3. AUDIO OUTPUT

| Part | Exact Product | Price (USD) | Notes |
|------|--------------|-------------|-------|
| USB DAC | **Sabrent AU-MMSA USB External Stereo Sound Adapter** | ~$8 | Class-compliant USB audio, works on Pi out of the box. The Pi 5's built-in 3.5mm jack is PWM-based and noisy — a USB DAC is strongly recommended for clean synth output. |
| Speaker | **Any small powered/active speaker with 3.5mm input** | ~$15-20 | A small desktop speaker works. Or a portable Bluetooth speaker with aux-in. Doesn't need to be fancy — it's a demo. |

**Subtotal: ~$23-28**

### 4. PHYSICAL INTERFACE (for the "reach goal" — standalone unit)

| Part | Exact Product | Price (USD) | Notes |
|------|--------------|-------------|-------|
| OLED display | **SSD1306 0.96" 128x64 I2C OLED** (any brand, I2C version) | ~$7-10 | Show current station freq, detected note, status. I2C = only 4 wires (VCC, GND, SDA, SCL). Use Pi GPIO pins 3 (SDA) and 5 (SCL). Library: `luma.oled` or `adafruit-circuitpython-ssd1306`. |
| Rotary encoder | **KY-040 Rotary Encoder Module** | ~$3-5 | For tuning FM frequency. Has built-in push button (for mode select). 3 signal pins + VCC + GND. |
| Push buttons | **Momentary tactile push buttons (6mm, pack of 10+)** | ~$3 | Start/stop, mute, etc. Wire to GPIO with internal pull-ups (no external resistors needed on Pi). |
| Breadboard | **Half-size solderless breadboard** | ~$4 | For prototyping. Can transfer to perfboard later if you want a cleaner build. |
| Jumper wires | **Male-to-female dupont jumper wires (40-pin ribbon)** | ~$4 | For connecting components to Pi GPIO header. |
| Project enclosure | **Any plastic project box ~200x120x60mm** | ~$8-12 | Or 3D print one. Needs holes for: USB-C power, USB DAC out, antenna SMA, OLED window, encoder shaft, buttons. |

**Subtotal: ~$29-38**

### 5. MISC

| Part | Exact Product | Price (USD) | Notes |
|------|--------------|-------------|-------|
| USB-A to USB-C adapter or hub | **Any USB-A hub** | ~$5-8 | Pi 5 has USB-A ports, but you'll have SDR dongle + USB DAC plugged in simultaneously. Both are USB-A so you should be fine with the Pi 5's built-in ports (2x USB 3.0 + 2x USB 2.0). Skip the hub unless you need more ports. |
| Extra microSD (backup) | Optional | ~$8 | Nice to have a backup OS image. |

**Subtotal: ~$0-8 (hub probably unnecessary)**

---

## TOTAL BOM

| Category | Cost |
|----------|------|
| Compute (Pi 5 + power + SD + cooling + case) | ~$97 |
| FM Reception (RTL-SDR V4 + antenna) | ~$42 |
| Audio Output (USB DAC + speaker) | ~$25 |
| Physical Interface (OLED + encoder + buttons + breadboard + wires + enclosure) | ~$34 |
| **TOTAL** | **~$198** |

Buffer of ~$52 for shipping, tax, or upgrades.

---

## WHERE TO BUY (US)

- **Pi 5 + official accessories**: rpilocator.com to find stock, or direct from Adafruit/SparkFun/PiShop.us
- **RTL-SDR Blog V4**: rtl-sdr.com (direct) or Amazon ("RTL-SDR Blog V4")
- **Small components** (OLED, encoder, buttons, wires): Amazon or Adafruit
- **USB DAC**: Amazon (Sabrent AU-MMSA)

---

## COMPATIBILITY NOTES

1. **Python on Pi 5**: Raspberry Pi OS (Bookworm, 64-bit) ships Python 3.11. Your code uses Python 3.10 but nothing version-specific — will work fine on 3.11.

2. **numpy/scipy on ARM64**: Both have pre-built wheels for aarch64 Linux. `pip install numpy scipy` just works on Pi 5. No compilation needed.

3. **sounddevice on Pi**: Needs PortAudio. `sudo apt install libportaudio2` then `pip install sounddevice`. Works with USB DAC out of the box via ALSA.

4. **RTL-SDR on Pi**: `sudo apt install rtl-sdr` gives you `rtl_fm`, `rtl_test`, etc. The V4 dongle is plug-and-play on Linux. May need to blacklist `dvb_usb_rtl28xxu` kernel module (standard RTL-SDR setup step).

5. **GPIO for controls**: `pip install RPi.GPIO` or `pip install gpiozero` (gpiozero is simpler). Both pre-installed on Raspberry Pi OS.

6. **OLED display**: `pip install luma.oled` — handles SSD1306 over I2C. Enable I2C via `sudo raspi-config`.

7. **Performance estimate**: Pi 5's Cortex-A76 @ 2.4GHz should handle the pipeline comfortably. Your DSP is numpy-based (BLAS-accelerated on ARM64). The bottleneck will be FFT in the harmonic pitch detector — 2048-point FFT at 44100/512 = ~86 frames/sec. Pi 5 can do thousands of 2048-point FFTs per second. You'll have headroom.

8. **Latency**: Your current pipeline latency is ~block_size/SR = 2048/44100 ≈ 46ms per block, plus Viterbi lookahead. FM demodulation via rtl_fm adds ~10-20ms. Total system latency should be under 100ms — fine for the demo.

---

## CODE ADAPTATION OVERVIEW (for reference, not for shopping)

Main changes needed from current live_musicbox.py:

1. **Input swap**: Replace sounddevice mic input with subprocess reading rtl_fm stdout (raw PCM bytes → numpy array). ~20 lines changed.

2. **Strip tkinter**: Replace GUI with headless main loop. Piano roll visualization → optional OLED display showing note name + frequency.

3. **Add GPIO controls**: Rotary encoder for FM tuning (changes rtl_fm frequency), buttons for start/stop/mute. ~50 lines new code.

4. **Synth output**: SineSynth already uses sounddevice output — just point it at the USB DAC device. May want to upgrade from pure sine to something more "ghostly" (FM synthesis, reverb, detuned oscillators). Optional but cool.

5. **Systemd service**: For headless auto-start on boot. ~10 line unit file.

Everything in digital_ear/ stays completely untouched.
