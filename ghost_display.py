#!/usr/bin/env python3
"""
GhostFM display rendering.

Retro-themed status display for GhostFM.
Runs in a daemon thread, reads pipeline state, draws to LCD at ~10 fps.

Top half: ghost sprite, logo, conf/gate, current note.
Bottom half: scrolling piano roll with rainbow-cycling colors.

Idle mode: ghost bounces around like the DVD logo with centered logo.

Mounted upside-down: MADCTL rotation = 90 degrees.
Supports the original Waveshare 1.3" ST7789 HAT and Linux framebuffer
devices such as the LCD-show MHS35 driver on /dev/fb1.
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

FONT_REGULAR_CANDIDATES = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]
FONT_BOLD_CANDIDATES = [
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


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


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a scalable font, falling back to Pillow's size-aware default."""
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_REGULAR_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


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
                 spi_speed_hz=40_000_000, backlight_pct=50):
        self.width = width
        self.height = height
        self.rotation = rotation
        self._backlight_pct = max(0, min(100, backlight_pct))
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

        # PWM backlight for camera-friendly brightness
        if self._backlight_pct >= 100:
            lgpio.gpio_claim_output(self._gpio, PIN_BL)
            lgpio.gpio_write(self._gpio, PIN_BL, 1)
        elif self._backlight_pct <= 0:
            lgpio.gpio_claim_output(self._gpio, PIN_BL)
            lgpio.gpio_write(self._gpio, PIN_BL, 0)
        else:
            lgpio.tx_pwm(self._gpio, PIN_BL, 1000, self._backlight_pct)

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


class FramebufferRGB565:
    """Linux framebuffer writer for LCD-show SPI displays."""

    def __init__(self, path="/dev/fb1", width=480, height=320, rotation=0, byte_order="little"):
        self.path = path
        self.width = width
        self.height = height
        self.rotation = rotation
        self.byte_order = byte_order
        self._fb = None

    def begin(self):
        self._fb = open(self.path, "r+b", buffering=0)

    def display(self, img: Image.Image):
        if self.rotation:
            img = img.rotate(-self.rotation, expand=True)
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.BICUBIC)
        if img.mode != "RGB":
            img = img.convert("RGB")

        raw = image_to_rgb565(img, byte_order=self.byte_order)
        expected = self.width * self.height * 2
        if len(raw) != expected:
            raise ValueError(f"framebuffer frame is {len(raw)} bytes, expected {expected}")

        self._fb.seek(0)
        self._fb.write(raw)

    def close(self):
        if self._fb:
            self._fb.close()
            self._fb = None


def image_to_rgb565(img: Image.Image, byte_order: str = "big") -> bytes:
    """Convert an RGB Pillow image to RGB565 bytes."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
    rgb565 = ((arr[:, 0] & 0xF8) << 8) | ((arr[:, 1] & 0xFC) << 3) | (arr[:, 2] >> 3)
    buf = np.empty(len(rgb565) * 2, dtype=np.uint8)
    if byte_order == "little":
        buf[0::2] = (rgb565 & 0xFF).astype(np.uint8)
        buf[1::2] = (rgb565 >> 8).astype(np.uint8)
    else:
        buf[0::2] = (rgb565 >> 8).astype(np.uint8)
        buf[1::2] = (rgb565 & 0xFF).astype(np.uint8)
    return buf.tobytes()


class GhostDisplay:
    """Drives a GhostFM status UI on ST7789 or framebuffer displays."""

    def __init__(
        self,
        ghost_sprite_path: Optional[str] = None,
        backlight_pct: int = 50,
        backend: str = "st7789",
        fbdev: str = "/dev/fb1",
        width: int = 240,
        height: int = 240,
        rotation: int = 0,
        byte_order: str = "little",
        normal_assets: bool = False,
        ui_scale: float = 1.0,
        asset_scale: float = 1.0,
    ):
        self._backlight_pct = backlight_pct
        self.backend = backend
        self.fbdev = fbdev
        self.width = width
        self.height = height
        self.rotation = rotation
        self.byte_order = byte_order
        self.normal_assets = normal_assets
        self.ui_scale = ui_scale
        self.asset_scale = asset_scale
        self.roll_top = int(self.height * 0.52)
        self.roll_height = self.height - self.roll_top
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
        self.edit_mode: bool = False
        self.edit_freq_str: str = ""   # e.g. "89.5" (no M suffix)
        self.edit_cursor: int = 0      # 0=tens, 1=ones, 2=tenths

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hw: Optional[object] = None
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
            ghost_name = "ghost_normal.png" if self.normal_assets else "ghost.png"
            ghost_sprite_path = os.path.join(base, "assets", ghost_name)
        self._ghost_path = ghost_sprite_path
        logo_name = "ghostfm_purple_normal.png" if self.normal_assets else "ghostfm_purple.png"
        self._logo_path = os.path.join(base, "assets", logo_name)

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

        if self.backend == "st7789":
            try:
                self._hw = ST7789Direct(rotation=90, backlight_pct=self._backlight_pct)
                self._hw.begin()
            except Exception as e:
                print(f"  LCD: ST7789 not available ({e}), display disabled")
                self._hw = None
                return
        elif self.backend == "fbdev":
            try:
                self._hw = FramebufferRGB565(
                    path=self.fbdev,
                    width=self.width,
                    height=self.height,
                    rotation=self.rotation,
                    byte_order=self.byte_order,
                )
                self._hw.begin()
            except Exception as e:
                print(f"  LCD: framebuffer not available ({e}), display disabled")
                self._hw = None
                return
        elif self.backend == "none":
            self._hw = None
        else:
            print(f"  LCD: unknown display backend {self.backend!r}, display disabled")
            return

        # load ghost sprite (normal + horizontally flipped)
        try:
            raw = Image.open(self._ghost_path).convert("RGBA")
            ghost_size = max(24, int(48 * self.asset_scale))
            self._ghost_img = raw.resize((ghost_size, ghost_size), Image.NEAREST)
            self._ghost_img_flipped = ImageOps.mirror(self._ghost_img)
        except Exception as e:
            print(f"  LCD: Ghost sprite not found ({e}), using fallback")
            self._ghost_img = self._make_fallback_ghost()
            self._ghost_img_flipped = ImageOps.mirror(self._ghost_img)

        # load logo (small for active UI)
        try:
            logo_raw = Image.open(self._logo_path).convert("RGBA")
            logo_h = max(16, int(28 * self.asset_scale))
            logo_w = int(logo_raw.width * logo_h / logo_raw.height)
            self._logo_img = logo_raw.resize((logo_w, logo_h), Image.LANCZOS)
            # larger logo for idle screen (centered)
            idle_h = max(24, int(40 * self.asset_scale))
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
                blank = Image.new("RGB", (self.width, self.height), BLACK)
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
        img = Image.new("RGB", (self.width, self.height), BLACK)

        # draw centered logo
        if self._logo_idle_img:
            lw, lh = self._logo_idle_img.size
            logo_x = (self.width - lw) // 2
            logo_y = (self.height - lh) // 2
            logo_layer = Image.new("RGB", (self.width, self.height), BLACK)
            logo_layer.paste(self._logo_idle_img, (logo_x, logo_y), self._logo_idle_img)
            logo_mask = self._logo_idle_img.split()[3]
            img.paste(logo_layer.crop((logo_x, logo_y, logo_x + lw, logo_y + lh)),
                      (logo_x, logo_y), logo_mask)
        else:
            draw = ImageDraw.Draw(img)
            font = _load_font(max(12, int(24 * self.ui_scale)), bold=True)
            draw.text((self.width * 0.25, self.height * 0.45), "GhostFM", fill=PURPLE, font=font)

        # DVD-bounce the ghost
        ghost_w, ghost_h = self._ghost_img.size if self._ghost_img else (48, 48)

        # update position
        self._bounce_x += self._bounce_vx
        self._bounce_y += self._bounce_vy

        # bounce off edges
        if self._bounce_x + ghost_w >= self.width:
            self._bounce_x = self.width - ghost_w
            self._bounce_vx = -abs(self._bounce_vx)
            self._bounce_dir_right = False
        elif self._bounce_x <= 0:
            self._bounce_x = 0
            self._bounce_vx = abs(self._bounce_vx)
            self._bounce_dir_right = True

        if self._bounce_y + ghost_h >= self.height:
            self._bounce_y = self.height - ghost_h
            self._bounce_vy = -abs(self._bounce_vy)
        elif self._bounce_y <= 0:
            self._bounce_y = 0
            self._bounce_vy = abs(self._bounce_vy)

        gx = int(self._bounce_x)
        gy = int(self._bounce_y)

        # pick the right facing sprite (flipped = facing right)
        sprite = self._ghost_img_flipped if self._bounce_dir_right else self._ghost_img

        if sprite:
            ghost_layer = Image.new("RGB", (self.width, self.height), BLACK)
            ghost_layer.paste(sprite, (gx, gy), sprite)
            mask = sprite.split()[3]
            img.paste(ghost_layer.crop((gx, gy, gx + ghost_w, gy + ghost_h)),
                      (gx, gy), mask)

        return img

    # ---- ACTIVE MODE ----

    def _draw_frame(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), BLACK)
        draw = ImageDraw.Draw(img)

        font_lg_size = max(10, int(20 * self.ui_scale))
        font_md_size = max(8, int(16 * self.ui_scale))
        font_sm_size = max(7, int(13 * self.ui_scale))
        font_lg = _load_font(font_lg_size, bold=True)
        font_md = _load_font(font_md_size)
        font_sm = _load_font(font_sm_size)

        # -- ghost sprite (bobbing animation) --
        bob_offset = int(3 * math.sin(self._frame_count * 0.4))
        margin = max(8, int(8 * self.asset_scale))
        ghost_x = margin
        ghost_y = max(8, int(10 * self.asset_scale)) + bob_offset

        ghost_w = ghost_h = max(48, int(48 * self.asset_scale))
        if self._ghost_img:
            ghost_w, ghost_h = self._ghost_img.size
            ghost_layer = Image.new("RGB", (self.width, self.height), BLACK)
            ghost_layer.paste(self._ghost_img, (ghost_x, ghost_y), self._ghost_img)
            mask = self._ghost_img.split()[3]
            img.paste(ghost_layer.crop((ghost_x, ghost_y, ghost_x + ghost_w, ghost_y + ghost_h)),
                      (ghost_x, ghost_y), mask)

        # -- title (logo or text fallback) --
        tight_gap = max(6, int(4 * self.ui_scale))
        title_x = margin + ghost_w + tight_gap
        logo_y = max(8, int(8 * self.asset_scale))
        logo_h = font_lg_size
        if self._logo_img:
            logo_x = title_x
            logo_layer = Image.new("RGB", (self.width, self.height), BLACK)
            logo_layer.paste(self._logo_img, (logo_x, logo_y), self._logo_img)
            logo_w, logo_h = self._logo_img.size
            logo_mask = self._logo_img.split()[3]
            img.paste(logo_layer.crop((logo_x, logo_y, logo_x + logo_w, logo_y + logo_h)),
                      (logo_x, logo_y), logo_mask)
        else:
            draw.text((title_x, logo_y), "GhostFM", fill=PURPLE, font=font_lg)

        # -- FM frequency (with edit mode support) --
        row_gap = tight_gap
        freq_y = logo_y + logo_h + row_gap
        if self.edit_mode:
            # show "EDIT" label + frequency with flashing cursor digit
            draw.text((title_x, freq_y), "EDIT ", fill=MUTED_RED, font=font_md)
            freq_str = self.edit_freq_str  # e.g. "89.5"
            blink_on = (int(time.monotonic() * 3) % 2) == 0

            # map cursor (0=tens, 1=ones, 2=tenths) to char index in freq_str
            # freq_str is like "89.5" (len 4) or "101.1" (len 5)
            digits_only = [i for i, c in enumerate(freq_str) if c.isdigit()]
            # cursor 0 = first digit, 1 = second digit, 2 = last digit (tenths)
            cursor_map = {0: digits_only[0] if len(digits_only) > 0 else 0,
                          1: digits_only[1] if len(digits_only) > 1 else 1,
                          2: digits_only[-1] if digits_only else len(freq_str)-1}
            cursor_char_idx = cursor_map.get(self.edit_cursor, 0)

            # draw each character, highlighting the cursor digit
            x_pos = title_x + max(46, int(self.width * 0.12))
            for i, ch in enumerate(freq_str):
                if i == cursor_char_idx and blink_on:
                    # draw highlighted (inverted)
                    bbox = font_md.getbbox(ch)
                    ch_w = bbox[2] - bbox[0]
                    draw.rectangle([x_pos - 1, freq_y - 1, x_pos + ch_w + 1, freq_y + font_md_size + 4], fill=BRIGHT)
                    draw.text((x_pos, freq_y), ch, fill=BLACK, font=font_md)
                else:
                    color = BRIGHT if i == cursor_char_idx else WHITE
                    draw.text((x_pos, freq_y), ch, fill=color, font=font_md)
                bbox = font_md.getbbox(ch)
                x_pos += bbox[2] - bbox[0] + 1
        else:
            draw.text((title_x, freq_y), f"FM {self.fm_freq}", fill=DIM, font=font_md)

        # -- separator --
        sep1 = freq_y + font_md_size + tight_gap
        draw.line([(margin, sep1), (self.width - margin, sep1)], fill=FAINT, width=1)

        # -- conf / gate --
        metrics_y = sep1 + tight_gap
        draw.text((margin, metrics_y), "conf", fill=DIM, font=font_sm)
        draw.text((margin + int(self.width * 0.16), metrics_y - 2), f"{self.conf_th:.1f}", fill=BRIGHT, font=font_md)
        draw.text((margin + int(self.width * 0.40), metrics_y), "gate", fill=DIM, font=font_sm)
        draw.text((margin + int(self.width * 0.54), metrics_y - 2), f"{self.rms_th:.3f}", fill=BRIGHT, font=font_md)

        # -- current note (color matches current rainbow hue) --
        note_y = metrics_y + font_md_size + tight_gap
        if self.note_name != "--":
            draw.text((margin, note_y), self.note_name, fill=self._current_color, font=font_lg)
            draw.text((margin + int(self.width * 0.18), note_y + 4), f"{self.freq_hz:.1f} Hz", fill=DIM, font=font_sm)
        else:
            draw.text((margin, note_y), "--", fill=DIM, font=font_lg)

        # -- flashing MUTED indicator (top-right corner) --
        if self.muted:
            blink_on = (int(time.monotonic() * 2) % 2) == 0
            if blink_on:
                draw.text((self.width - max(62, int(self.width * 0.18)), margin), "MUTED", fill=MUTED_RED, font=font_sm)

        # -- mode --
        mode_color = BRIGHT if self.mode == "GHOST" else WHITE
        if self.muted:
            mode_color = FAINT
        draw.text((self.width - max(80, int(self.width * 0.22)), note_y + 2), self.mode, fill=mode_color, font=font_sm)

        # -- separator (top/bottom half boundary) --
        min_roll_height = max(52, int(34 * self.ui_scale))
        desired_roll_top = note_y + font_lg_size + tight_gap
        self.roll_top = min(max(int(self.height * 0.52), desired_roll_top), self.height - min_roll_height)
        self.roll_height = self.height - self.roll_top
        draw.line([(margin, self.roll_top - 4), (self.width - margin, self.roll_top - 4)], fill=FAINT, width=1)

        # -- piano roll (bottom half) --
        self._draw_roll(draw, img)

        return img

    def _draw_roll(self, draw: ImageDraw.Draw, img: Image.Image):
        """Draw the scrolling piano roll in the bottom half."""
        now = time.monotonic()
        key_h = self.roll_height / NUM_KEYS
        px_per_sec = self.width / ROLL_SECONDS

        # dim octave grid lines
        for i in range(NUM_KEYS):
            midi = MIDI_HI - 1 - i
            if midi % 12 == 0:
                y = self.roll_top + int(i * key_h)
                draw.line([(0, y), (self.width, y)], fill=(25, 15, 35), width=1)

        # draw completed notes (each has its own stamped color)
        for note in self._roll_notes:
            if note.midi < MIDI_LO or note.midi >= MIDI_HI:
                continue

            x_start = int(self.width - (now - note.start_time) * px_per_sec)
            x_end = int(self.width - (now - note.end_time) * px_per_sec)

            if x_end < 0 or x_start > self.width:
                continue

            row = MIDI_HI - 1 - note.midi
            y = self.roll_top + int(row * key_h) + 1
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
            x_start = int(self.width - (now - self._live_start) * px_per_sec)
            x_start = max(0, x_start)

            row = MIDI_HI - 1 - self._live_midi
            y = self.roll_top + int(row * key_h) + 1
            h = max(int(key_h) - 2, 1)

            draw.rectangle([x_start, y, self.width, y + h], fill=self._live_color)

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
