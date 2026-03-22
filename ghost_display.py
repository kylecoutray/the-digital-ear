#!/usr/bin/env python3
"""
GhostFM LCD Display — Waveshare 1.3" LCD HAT (240x240, ST7789)

Retro-themed status display for GhostFM.
Runs in a daemon thread, reads pipeline state, draws to LCD at ~10 fps.

Mounted upside-down: MADCTL rotation = 0xC0 (180 degrees).
Uses direct SPI via spidev + lgpio (Pi 5 compatible).
Graceful fallback: if SPI/lgpio unavailable, prints warning and no-ops.
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# display dimensions
WIDTH = 240
HEIGHT = 240

# color scheme (retro purple/green on black)
BLACK = (0, 0, 0)
PURPLE = (200, 100, 255)       # more saturated for RGB565 display
GREEN = (118, 255, 3)          # #76ff03
DIM_PURPLE = (100, 70, 160)
DIM_GREEN = (60, 130, 2)
WHITE = (220, 220, 220)

# ST7789 GPIO pins (Waveshare 1.3" LCD HAT)
PIN_DC = 25
PIN_RST = 27
PIN_BL = 24


class ST7789Direct:
    """Minimal ST7789 driver using spidev + lgpio (Pi 5 compatible)."""

    def __init__(self, width=240, height=240, rotation=90,
                 spi_speed_hz=40_000_000):
        self.width = width
        self.height = height
        self.rotation = rotation
        self._spi = None
        self._gpio = None
        # ST7789 has 240x320 framebuffer; 240x240 display needs offset
        self._col_offset = 0
        self._row_offset = 0

    def begin(self):
        import spidev
        import lgpio

        # GPIO
        self._lgpio = lgpio
        self._gpio = lgpio.gpiochip_open(4)  # Pi 5 = chip 4
        lgpio.gpio_claim_output(self._gpio, PIN_DC)
        lgpio.gpio_claim_output(self._gpio, PIN_RST)
        lgpio.gpio_claim_output(self._gpio, PIN_BL)

        # backlight on
        lgpio.gpio_write(self._gpio, PIN_BL, 1)

        # hardware reset
        lgpio.gpio_write(self._gpio, PIN_RST, 1)
        time.sleep(0.1)
        lgpio.gpio_write(self._gpio, PIN_RST, 0)
        time.sleep(0.1)
        lgpio.gpio_write(self._gpio, PIN_RST, 1)
        time.sleep(0.1)

        # SPI
        self._spi = spidev.SpiDev(0, 0)
        self._spi.max_speed_hz = 40_000_000
        self._spi.mode = 0

        # init sequence
        self._cmd(0x01)   # software reset
        time.sleep(0.15)
        self._cmd(0x11)   # sleep out
        time.sleep(0.15)
        self._cmd(0x3A); self._data(0x05)   # 16-bit color RGB565
        # MADCTL for rotation + framebuffer offsets (240x320 -> 240x240)
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
        self._cmd(0x21)   # inversion on
        self._cmd(0x29)   # display on
        time.sleep(0.05)

    def display(self, img: Image.Image):
        """Push a PIL RGB image to the display."""
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # set window (with framebuffer offsets)
        x0 = self._col_offset
        x1 = self._col_offset + self.width - 1
        y0 = self._row_offset
        y1 = self._row_offset + self.height - 1
        self._cmd(0x2A)
        self._data([(x0 >> 8) & 0xFF, x0 & 0xFF, (x1 >> 8) & 0xFF, x1 & 0xFF])
        self._cmd(0x2B)
        self._data([(y0 >> 8) & 0xFF, y0 & 0xFF, (y1 >> 8) & 0xFF, y1 & 0xFF])
        self._cmd(0x2C)

        # convert to RGB565 (numpy vectorized — fast + accurate)
        arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
        rgb565 = ((arr[:, 0] & 0xF8) << 8) | ((arr[:, 1] & 0xFC) << 3) | (arr[:, 2] >> 3)
        # big-endian byte order for SPI
        buf = np.empty(len(rgb565) * 2, dtype=np.uint8)
        buf[0::2] = (rgb565 >> 8).astype(np.uint8)
        buf[1::2] = (rgb565 & 0xFF).astype(np.uint8)
        raw = buf.tobytes()

        # send in chunks (spidev limit)
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
    """Drives the Waveshare 1.3" LCD HAT with a retro GhostFM status UI.

    Usage:
        display = GhostDisplay()
        display.start()
        # update state as needed:
        display.conf_th = 5.0
        display.rms_th = 0.003
        display.note_name = "A4"
        display.freq_hz = 440.0
        # ...
        display.stop()
    """

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

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hw: Optional[ST7789Direct] = None
        self._ghost_img: Optional[Image.Image] = None
        self._logo_img: Optional[Image.Image] = None
        self._frame_count = 0

        # resolve asset paths
        base = os.path.dirname(os.path.abspath(__file__))
        if ghost_sprite_path is None:
            ghost_sprite_path = os.path.join(base, "assets", "ghost.png")
        self._ghost_path = ghost_sprite_path
        self._logo_path = os.path.join(base, "assets", "ghostfm_purple.png")

    def start(self):
        """Initialize display hardware and start render thread."""
        if not _HAS_PIL:
            print("  LCD: Pillow not available, display disabled")
            return

        # try to init ST7789 via direct SPI
        try:
            self._hw = ST7789Direct(rotation=90)
            self._hw.begin()
        except Exception as e:
            print(f"  LCD: ST7789 not available ({e}), display disabled")
            self._hw = None
            return

        # load ghost sprite
        try:
            raw = Image.open(self._ghost_path).convert("RGBA")
            self._ghost_img = raw.resize((48, 48), Image.NEAREST)
        except Exception as e:
            print(f"  LCD: Ghost sprite not found ({e}), using fallback")
            self._ghost_img = self._make_fallback_ghost()

        # load logo (374x127 -> fit in ~168x28 px, next to ghost)
        try:
            logo_raw = Image.open(self._logo_path).convert("RGBA")
            # scale to ~28px tall, preserve aspect ratio
            logo_h = 28
            logo_w = int(logo_raw.width * logo_h / logo_raw.height)
            self._logo_img = logo_raw.resize((logo_w, logo_h), Image.LANCZOS)
        except Exception as e:
            print(f"  LCD: Logo not found ({e}), using text fallback")
            self._logo_img = None

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print("  LCD: Display started")

    def stop(self):
        """Stop render thread and blank display."""
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
        """Main render loop, ~10 fps."""
        while self._running:
            try:
                frame = self._draw_frame()
                if self._hw:
                    self._hw.display(frame)
                self._frame_count += 1
            except Exception:
                pass
            time.sleep(0.1)  # ~10 fps

    def _draw_frame(self) -> Image.Image:
        """Draw one frame of the status UI."""
        img = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
        draw = ImageDraw.Draw(img)

        # load font (PIL default bitmap font)
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
            # composite onto a black background to preserve full color
            ghost_layer = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
            ghost_layer.paste(self._ghost_img, (ghost_x, ghost_y), self._ghost_img)
            # merge only the ghost area
            mask = self._ghost_img.split()[3]  # alpha channel
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
        draw.text((64, 42), f"FM {self.fm_freq}", fill=DIM_PURPLE, font=font_sm)

        # -- separator line --
        draw.line([(8, 68), (232, 68)], fill=DIM_PURPLE, width=1)

        # -- conf / gate values --
        draw.text((8, 76), "conf", fill=DIM_GREEN, font=font_sm)
        draw.text((50, 74), f"{self.conf_th:.1f}", fill=GREEN, font=font_md)

        draw.text((120, 76), "gate", fill=DIM_GREEN, font=font_sm)
        draw.text((162, 74), f"{self.rms_th:.3f}", fill=GREEN, font=font_md)

        # -- current note --
        if self.note_name != "--":
            draw.text((8, 98), self.note_name, fill=PURPLE, font=font_lg)
            draw.text((60, 100), f"{self.freq_hz:.1f} Hz", fill=DIM_PURPLE, font=font_sm)
        else:
            draw.text((8, 98), "--", fill=DIM_PURPLE, font=font_lg)

        # -- mode / mute indicator --
        mode_color = GREEN if self.mode == "GHOST" else PURPLE
        mode_text = self.mode
        if self.muted:
            mode_text += " MUTED"
            mode_color = DIM_PURPLE
        draw.text((170, 98), mode_text, fill=mode_color, font=font_sm)

        # -- bottom separator (top-half boundary) --
        draw.line([(8, 120), (232, 120)], fill=DIM_PURPLE, width=1)

        return img

    @staticmethod
    def _make_fallback_ghost() -> Image.Image:
        """Draw a simple pixel-art ghost if no sprite file is found."""
        img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        ghost_color = (179, 136, 255, 255)

        # simple ghost body (rounded top, wavy bottom)
        # head
        draw.ellipse([12, 4, 36, 28], fill=ghost_color)
        # body
        draw.rectangle([12, 16, 36, 38], fill=ghost_color)
        # wavy bottom (3 bumps)
        for i in range(3):
            x = 12 + i * 8
            draw.ellipse([x, 34, x + 8, 44], fill=ghost_color)
        # eyes
        draw.rectangle([18, 14, 22, 20], fill=BLACK)
        draw.rectangle([26, 14, 30, 20], fill=BLACK)
        # headphone band
        draw.arc([10, 2, 38, 22], 180, 0, fill=(118, 255, 3, 255), width=2)
        # ear cups
        draw.rectangle([8, 12, 14, 22], fill=(118, 255, 3, 255))
        draw.rectangle([34, 12, 40, 22], fill=(118, 255, 3, 255))

        return img
