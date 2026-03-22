#!/usr/bin/env python3
"""
GhostFM LCD Display — Waveshare 1.3" LCD HAT (240x240, ST7789)

Retro-themed status display for GhostFM.
Runs in a daemon thread, reads pipeline state, draws to LCD at ~10 fps.

Top half: ghost sprite, logo, conf/gate, current note.
Bottom half: scrolling piano roll with rainbow-cycling colors.

Idle mode: ghost bounces around like the DVD logo with centered logo.

Mounted upside-down: MADCTL rotation = 90 degrees.
Uses direct SPI via spidev + lgpio (Pi 5 compatible).
Graceful fallback: if SPI/lgpio unavailable, prints warning and no-ops.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# display dimensions
WIDTH = 240
HEIGHT = 240

# piano roll region (bottom half)
ROLL_TOP = 124       # just below the separator at y=120
ROLL_HEIGHT = HEIGHT - ROLL_TOP
MIDI_LO = 48         # C3
MIDI_HI = 84         # C6
NUM_KEYS = MIDI_HI - MIDI_LO
ROLL_SECONDS = 6.0   # seconds of history visible

# color scheme (purple on black, tuned for RGB565 display)
BLACK = (0, 0, 0)
PURPLE = (255, 0, 255)         # full magenta
BRIGHT = (200, 80, 255)        # accent purple for values
DIM = (120, 40, 180)           # dim purple for labels
FAINT = (70, 20, 110)          # very dim for separators
WHITE = (220, 220, 220)
MUTED_RED = (255, 40, 40)      # flashing muted indicator

# Paradromics aura gradient (orange -> blue -> purple -> red)
AURA = [
    (240, 140, 30), (210, 100, 20), (80, 50, 160),
    (30, 70, 210), (50, 100, 240), (70, 80, 220),
    (140, 50, 180), (210, 55, 65), (180, 40, 35),
]

# ST7789 GPIO pins (Waveshare 1.3" LCD HAT)
PIN_DC = 25
PIN_RST = 27
PIN_BL = 24


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linearly interpolate between two RGB tuples."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _aura_color(t: float) -> tuple:
    """Get a color from the AURA gradient at position t (0..1)."""
    t = t % 1.0
    n = len(AURA) - 1
    idx = int(t * n)
    frac = (t * n) - idx
    if idx >= n:
        idx = n - 1
        frac = 1.0
    return _lerp_color(AURA[idx], AURA[idx + 1], frac)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    """Convert HSV (0..1) to RGB (0..255)."""
    if s == 0.0:
        c = int(v * 255)
        return (c, c, c)
    h6 = h * 6.0
    i = int(h6)
    f = h6 - i
    p = int(v * (1.0 - s) * 255)
    q = int(v * (1.0 - s * f) * 255)
    t_val = int(v * (1.0 - s * (1.0 - f)) * 255)
    v_int = int(v * 255)
    i = i % 6
    if i == 0: return (v_int, t_val, p)
    if i == 1: return (q, v_int, p)
    if i == 2: return (p, v_int, t_val)
    if i == 3: return (p, q, v_int)
    if i == 4: return (t_val, p, v_int)
    return (v_int, p, q)


@dataclass
class RollNote:
    """A note event for the piano roll display."""
    midi: int
    start_time: float
    end_time: float  # 0 = still playing
    color: tuple = (255, 255, 255)  # RGB color at time of creation


class ST7789Direct:
    """Minimal ST7789 driver using spidev + lgpio (Pi 5 compatible)."""

    def __init__(self, width=240, height=240, rotation=90,
                 spi_speed_hz=40_000_000):
        self.width = width
        self.height = height
        self.rotation = rotation
        self._spi = None
        self._gpio = None
        self._col_offset = 0
        self._row_offset = 0

    def begin(self):
        import spidev
        import lgpio

        self._lgpio = lgpio
        self._gpio = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(self._gpio, PIN_DC)
        lgpio.gpio_claim_output(self._gpio, PIN_RST)
        lgpio.gpio_claim_output(self._gpio, PIN_BL)

        lgpio.gpio_write(self._gpio, PIN_BL, 1)

        lgpio.gpio_write(self._gpio, PIN_RST, 1)
        time.sleep(0.1)
        lgpio.gpio_write(self._gpio, PIN_RST, 0)
        time.sleep(0.1)
        lgpio.gpio_write(self._gpio, PIN_RST, 1)
        time.sleep(0.1)

        self._spi = spidev.SpiDev(0, 0)
        self._spi.max_speed_hz = 40_000_000
        self._spi.mode = 0

        self._cmd(0x01)
        time.sleep(0.15)
        self._cmd(0x11)
        time.sleep(0.15)
        self._cmd(0x3A); self._data(0x05)
        if self.rotation == 180:
            self._cmd(0x36); self._data(0xC0)
            self._col_offset = 0; self._row_offset = 80
        elif self.rotation == 90:
            self._cmd(0x36); self._data(0xA0)
            self._col_offset = 80; self._row_offset = 0
        elif self.rotation == 270:
            self._cmd(0x36); self._data(0x60)
            self._col_offset = 0; self._row_offset = 0
        else:
            self._cmd(0x36); self._data(0x00)
            self._col_offset = 0; self._row_offset = 0
        self._cmd(0x21)
        self._cmd(0x29)
        time.sleep(0.05)

    def display(self, img: Image.Image):
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height))
        if img.mode != "RGB":
            img = img.convert("RGB")

        x0 = self._col_offset
        x1 = self._col_offset + self.width - 1
        y0 = self._row_offset
        y1 = self._row_offset + self.height - 1
        self._cmd(0x2A)
        self._data([(x0 >> 8) & 0xFF, x0 & 0xFF, (x1 >> 8) & 0xFF, x1 & 0xFF])
        self._cmd(0x2B)
        self._data([(y0 >> 8) & 0xFF, y0 & 0xFF, (y1 >> 8) & 0xFF, y1 & 0xFF])
        self._cmd(0x2C)

        arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
        rgb565 = ((arr[:, 0] & 0xF8) << 8) | ((arr[:, 1] & 0xFC) << 3) | (arr[:, 2] >> 3)
        buf = np.empty(len(rgb565) * 2, dtype=np.uint8)
        buf[0::2] = (rgb565 >> 8).astype(np.uint8)
        buf[1::2] = (rgb565 & 0xFF).astype(np.uint8)
        raw = buf.tobytes()

        self._lgpio.gpio_write(self._gpio, PIN_DC, 1)
        for i in range(0, len(raw), 4096):
            self._spi.writebytes(raw[i:i+4096])

    def close(self):
        if self._spi:
            self._spi.close()
        if self._gpio is not None:
            self._lgpio.gpiochip_close(self._gpio)

    def _cmd(self, c):
        self._lgpio.gpio_write(self._gpio, PIN_DC, 0)
        self._spi.writebytes([c])

    def _data(self, d):
        self._lgpio.gpio_write(self._gpio, PIN_DC, 1)
        if isinstance(d, list):
            self._spi.writebytes(d)
        else:
            self._spi.writebytes([d])


class GhostDisplay:
    """Drives the Waveshare 1.3" LCD HAT with a retro GhostFM status UI."""

    def __init__(self, ghost_sprite_path: Optional[str] = None):
        # shared state (written by main thread, read by display thread)
        self.conf_th: float = 1.0
        self.rms_th: float = 0.003
        self.note_name: str = "--"
        self.freq_hz: float = 0.0
        self.note_count: int = 0
        self.fm_freq: str = "89.9M"
        self.mode: str = "GHOST"
        self.muted: bool = False
        self.current_midi: int = 0
        self.paused: bool = True   # starts paused (idle screen)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hw: Optional[ST7789Direct] = None
        self._ghost_img: Optional[Image.Image] = None
        self._ghost_img_flipped: Optional[Image.Image] = None
        self._logo_img: Optional[Image.Image] = None
        self._logo_idle_img: Optional[Image.Image] = None  # larger logo for idle screen
        self._frame_count = 0

        # piano roll note history
        self._roll_notes: deque[RollNote] = deque(maxlen=200)
        self._live_midi: int = 0
        self._live_start: float = 0.0
        self._roll_time: float = 0.0

        # resolve asset paths
        base = os.path.dirname(os.path.abspath(__file__))
        if ghost_sprite_path is None:
            ghost_sprite_path = os.path.join(base, "assets", "ghost.png")
        self._ghost_path = ghost_sprite_path
        self._logo_path = os.path.join(base, "assets", "ghostfm_purple.png")

        # rainbow hue cycling
        self._hue_cycle_rate = 0.1
        self._start_time = time.monotonic()
        self._current_color = (255, 255, 255)
        self._live_color = (255, 255, 255)

        # DVD bounce state for idle mode
        self._bounce_x: float = 60.0
        self._bounce_y: float = 80.0
        self._bounce_vx: float = 1.5   # pixels per frame
        self._bounce_vy: float = 1.0
        self._bounce_dir_right: bool = True  # for horizontal flip

    def _get_hue_color(self) -> tuple:
        elapsed = time.monotonic() - self._start_time
        hue = (elapsed * self._hue_cycle_rate) % 1.0
        return _hsv_to_rgb(hue, 0.9, 1.0)

    def push_note(self, midi: int):
        """Called by main thread when the current note changes."""
        now = time.monotonic()
        self._roll_time = now
        self._current_color = self._get_hue_color()

        if midi == self._live_midi:
            return

        if self._live_midi > 0:
            self._roll_notes.append(RollNote(
                midi=self._live_midi,
                start_time=self._live_start,
                end_time=now,
                color=self._live_color,
            ))

        self._live_midi = midi
        self._live_start = now if midi > 0 else 0.0
        self._live_color = self._current_color

    def start(self):
        if not _HAS_PIL:
            print("  LCD: Pillow not available, display disabled")
            return

        try:
            self._hw = ST7789Direct(rotation=90)
            self._hw.begin()
        except Exception as e:
            print(f"  LCD: ST7789 not available ({e}), display disabled")
            self._hw = None
            return

        # load ghost sprite (normal + horizontally flipped)
        try:
            raw = Image.open(self._ghost_path).convert("RGBA")
            self._ghost_img = raw.resize((48, 48), Image.NEAREST)
            self._ghost_img_flipped = ImageOps.mirror(self._ghost_img)
        except Exception as e:
            print(f"  LCD: Ghost sprite not found ({e}), using fallback")
            self._ghost_img = self._make_fallback_ghost()
            self._ghost_img_flipped = ImageOps.mirror(self._ghost_img)

        # load logo (small for active UI)
        try:
            logo_raw = Image.open(self._logo_path).convert("RGBA")
            logo_h = 28
            logo_w = int(logo_raw.width * logo_h / logo_raw.height)
            self._logo_img = logo_raw.resize((logo_w, logo_h), Image.LANCZOS)
            # larger logo for idle screen (centered)
            idle_h = 40
            idle_w = int(logo_raw.width * idle_h / logo_raw.height)
            self._logo_idle_img = logo_raw.resize((idle_w, idle_h), Image.LANCZOS)
        except Exception as e:
            print(f"  LCD: Logo not found ({e}), using text fallback")
            self._logo_img = None
            self._logo_idle_img = None

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print("  LCD: Display started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._hw:
            try:
                blank = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
                self._hw.display(blank)
                self._hw.close()
            except Exception:
                pass

    def _render_loop(self):
        while self._running:
            try:
                if self.paused:
                    frame = self._draw_idle()
                else:
                    frame = self._draw_frame()
                if self._hw:
                    self._hw.display(frame)
                self._frame_count += 1
            except Exception:
                pass
            time.sleep(0.1)

    # ---- IDLE MODE (DVD bounce) ----

    def _draw_idle(self) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), BLACK)

        # draw centered logo
        if self._logo_idle_img:
            lw, lh = self._logo_idle_img.size
            logo_x = (WIDTH - lw) // 2
            logo_y = (HEIGHT - lh) // 2
            logo_layer = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
            logo_layer.paste(self._logo_idle_img, (logo_x, logo_y), self._logo_idle_img)
            logo_mask = self._logo_idle_img.split()[3]
            img.paste(logo_layer.crop((logo_x, logo_y, logo_x + lw, logo_y + lh)),
                      (logo_x, logo_y), logo_mask)
        else:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((60, 110), "GhostFM", fill=PURPLE, font=font)

        # DVD-bounce the ghost
        ghost_w, ghost_h = 48, 48

        # update position
        self._bounce_x += self._bounce_vx
        self._bounce_y += self._bounce_vy

        # bounce off edges
        if self._bounce_x + ghost_w >= WIDTH:
            self._bounce_x = WIDTH - ghost_w
            self._bounce_vx = -abs(self._bounce_vx)
            self._bounce_dir_right = False
        elif self._bounce_x <= 0:
            self._bounce_x = 0
            self._bounce_vx = abs(self._bounce_vx)
            self._bounce_dir_right = True

        if self._bounce_y + ghost_h >= HEIGHT:
            self._bounce_y = HEIGHT - ghost_h
            self._bounce_vy = -abs(self._bounce_vy)
        elif self._bounce_y <= 0:
            self._bounce_y = 0
            self._bounce_vy = abs(self._bounce_vy)

        gx = int(self._bounce_x)
        gy = int(self._bounce_y)

        # pick the right facing sprite
        sprite = self._ghost_img if self._bounce_dir_right else self._ghost_img_flipped

        if sprite:
            ghost_layer = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
            ghost_layer.paste(sprite, (gx, gy), sprite)
            mask = sprite.split()[3]
            img.paste(ghost_layer.crop((gx, gy, gx + ghost_w, gy + ghost_h)),
                      (gx, gy), mask)

        return img

    # ---- ACTIVE MODE ----

    def _draw_frame(self) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
        draw = ImageDraw.Draw(img)

        try:
            font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20)
            font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
            font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        except Exception:
            font_lg = ImageFont.load_default()
            font_md = font_lg
            font_sm = font_lg

        # -- ghost sprite (bobbing animation) --
        bob_offset = int(3 * math.sin(self._frame_count * 0.4))
        ghost_x = 8
        ghost_y = 10 + bob_offset

        if self._ghost_img:
            ghost_layer = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
            ghost_layer.paste(self._ghost_img, (ghost_x, ghost_y), self._ghost_img)
            mask = self._ghost_img.split()[3]
            img.paste(ghost_layer.crop((ghost_x, ghost_y, ghost_x + 48, ghost_y + 48)),
                      (ghost_x, ghost_y), mask)

        # -- title (logo or text fallback) --
        if self._logo_img:
            logo_x = 64
            logo_y = 12
            logo_layer = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
            logo_layer.paste(self._logo_img, (logo_x, logo_y), self._logo_img)
            logo_w, logo_h = self._logo_img.size
            logo_mask = self._logo_img.split()[3]
            img.paste(logo_layer.crop((logo_x, logo_y, logo_x + logo_w, logo_y + logo_h)),
                      (logo_x, logo_y), logo_mask)
        else:
            draw.text((64, 14), "GhostFM", fill=PURPLE, font=font_lg)

        # -- FM frequency --
        draw.text((64, 42), f"FM {self.fm_freq}", fill=DIM, font=font_sm)

        # -- separator --
        draw.line([(8, 68), (232, 68)], fill=FAINT, width=1)

        # -- conf / gate --
        draw.text((8, 76), "conf", fill=DIM, font=font_sm)
        draw.text((50, 74), f"{self.conf_th:.1f}", fill=BRIGHT, font=font_md)
        draw.text((120, 76), "gate", fill=DIM, font=font_sm)
        draw.text((162, 74), f"{self.rms_th:.3f}", fill=BRIGHT, font=font_md)

        # -- current note (color matches current rainbow hue) --
        if self.note_name != "--":
            draw.text((8, 98), self.note_name, fill=self._current_color, font=font_lg)
            draw.text((60, 100), f"{self.freq_hz:.1f} Hz", fill=DIM, font=font_sm)
        else:
            draw.text((8, 98), "--", fill=DIM, font=font_lg)

        # -- mode + muted indicator --
        mode_color = BRIGHT if self.mode == "GHOST" else PURPLE
        mode_text = self.mode
        draw.text((170, 86), mode_text, fill=mode_color, font=font_sm)

        # flashing MUTED indicator (slow blink ~1Hz)
        if self.muted:
            blink_on = (int(time.monotonic() * 2) % 2) == 0
            if blink_on:
                draw.text((162, 102), "MUTED", fill=MUTED_RED, font=font_sm)

        # -- separator (top/bottom half boundary) --
        draw.line([(8, 120), (232, 120)], fill=FAINT, width=1)

        # -- piano roll (bottom half) --
        self._draw_roll(draw, img)

        return img

    def _draw_roll(self, draw: ImageDraw.Draw, img: Image.Image):
        """Draw the scrolling piano roll in the bottom half."""
        now = time.monotonic()
        key_h = ROLL_HEIGHT / NUM_KEYS
        px_per_sec = WIDTH / ROLL_SECONDS

        # dim octave grid lines
        for i in range(NUM_KEYS):
            midi = MIDI_HI - 1 - i
            if midi % 12 == 0:
                y = ROLL_TOP + int(i * key_h)
                draw.line([(0, y), (WIDTH, y)], fill=(25, 15, 35), width=1)

        # draw completed notes (each has its own stamped color)
        for note in self._roll_notes:
            if note.midi < MIDI_LO or note.midi >= MIDI_HI:
                continue

            x_start = int(WIDTH - (now - note.start_time) * px_per_sec)
            x_end = int(WIDTH - (now - note.end_time) * px_per_sec)

            if x_end < 0 or x_start > WIDTH:
                continue

            row = MIDI_HI - 1 - note.midi
            y = ROLL_TOP + int(row * key_h) + 1
            h = max(int(key_h) - 2, 1)

            color = note.color

            # fade with age
            age = now - note.end_time
            if age > 2.0:
                t = min((age - 2.0) / 3.0, 0.85)
                color = _lerp_color(color, (15, 10, 20), t)

            draw.rectangle([x_start, y, x_end, y + h], fill=color)

        # draw live note (extends to right edge, uses current live color)
        if self._live_midi > 0 and MIDI_LO <= self._live_midi < MIDI_HI:
            x_start = int(WIDTH - (now - self._live_start) * px_per_sec)
            x_start = max(0, x_start)

            row = MIDI_HI - 1 - self._live_midi
            y = ROLL_TOP + int(row * key_h) + 1
            h = max(int(key_h) - 2, 1)

            draw.rectangle([x_start, y, WIDTH, y + h], fill=self._live_color)

        # prune old notes
        cutoff = now - ROLL_SECONDS * 2
        while self._roll_notes and self._roll_notes[0].end_time < cutoff:
            self._roll_notes.popleft()

    @staticmethod
    def _make_fallback_ghost() -> Image.Image:
        img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        ghost_color = (255, 0, 255, 255)

        draw.ellipse([12, 4, 36, 28], fill=ghost_color)
        draw.rectangle([12, 16, 36, 38], fill=ghost_color)
        for i in range(3):
            x = 12 + i * 8
            draw.ellipse([x, 34, x + 8, 44], fill=ghost_color)
        draw.rectangle([18, 14, 22, 20], fill=BLACK)
        draw.rectangle([26, 14, 30, 20], fill=BLACK)
        draw.arc([10, 2, 38, 22], 180, 0, fill=(200, 80, 255, 255), width=2)
        draw.rectangle([8, 12, 14, 22], fill=(200, 80, 255, 255))
        draw.rectangle([34, 12, 40, 22], fill=(200, 80, 255, 255))

        return img
