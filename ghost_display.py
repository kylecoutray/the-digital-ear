#!/usr/bin/env python3
"""
GhostFM LCD Display — Waveshare 1.3" LCD HAT (240x240, ST7789)

Retro-themed status display for GhostFM.
Runs in a daemon thread, reads pipeline state, draws to LCD at ~10 fps.

Mounted upside-down: display rotation = 180 degrees.
Graceful fallback: if st7789/SPI unavailable, prints warning and no-ops.
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Optional

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
PURPLE = (179, 136, 255)       # #b388ff
GREEN = (118, 255, 3)          # #76ff03
DIM_PURPLE = (100, 70, 160)
DIM_GREEN = (60, 130, 2)
WHITE = (220, 220, 220)


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
        self._display = None
        self._ghost_img: Optional[Image.Image] = None
        self._frame_count = 0

        # resolve ghost sprite path
        if ghost_sprite_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            ghost_sprite_path = os.path.join(base, "assets", "ghost.png")
        self._ghost_path = ghost_sprite_path

    def start(self):
        """Initialize display hardware and start render thread."""
        if not _HAS_PIL:
            print("  LCD: Pillow not available, display disabled")
            return

        # try to init ST7789
        try:
            import st7789
            self._display = st7789.ST7789(
                height=240,
                width=240,
                rotation=180,
                port=0,
                cs=0,         # CE0 = GPIO 8
                dc=25,
                backlight=24,
                rst=27,
                spi_speed_hz=40_000_000,
            )
            self._display.begin()
        except Exception as e:
            print(f"  LCD: ST7789 not available ({e}), display disabled")
            self._display = None
            return

        # load ghost sprite
        try:
            raw = Image.open(self._ghost_path).convert("RGBA")
            self._ghost_img = raw.resize((48, 48), Image.NEAREST)
        except Exception as e:
            print(f"  LCD: Ghost sprite not found ({e}), using fallback")
            self._ghost_img = self._make_fallback_ghost()

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print("  LCD: Display started")

    def stop(self):
        """Stop render thread and blank display."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._display:
            try:
                # blank the screen
                blank = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
                self._display.display(blank)
            except Exception:
                pass

    def _render_loop(self):
        """Main render loop, ~10 fps."""
        while self._running:
            try:
                frame = self._draw_frame()
                if self._display:
                    self._display.display(frame)
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
            img.paste(self._ghost_img, (ghost_x, ghost_y), self._ghost_img)

        # -- title --
        draw.text((64, 14), "GhostFM", fill=PURPLE, font=font_lg)

        # -- FM frequency --
        draw.text((64, 38), f"FM {self.fm_freq}", fill=DIM_PURPLE, font=font_sm)

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
