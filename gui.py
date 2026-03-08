#!/usr/bin/env python3
"""
The Digital Ear — tkinter GUI (no extra deps).
"""
import datetime
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont

# Colour palette
BG         = "#0a0a0a"
BG_CARD    = "#141416"
BG_FIELD   = "#1c1c20"
FG         = "#e8e0d4"
FG_DIM     = "#7a7a82"
FG_MUTED   = "#9a9aa0"
ACCENT     = "#c8402d"
BORDER     = "#2a2a2e"
GEN_BG     = "#e8e0d4"
GEN_FG     = "#0a0a0a"
GEN_HV     = "#ffffff"
GEN_DIS    = "#3a3a3e"
ENTRY_BG   = "#18181c"
ENTRY_FG   = "#e8e0d4"
LOG_BG     = "#0e0e10"
LOG_FG     = "#9a9aa0"
STAT_FG    = "#d0ccc4"
CMD_FG     = "#6e6e74"
ERR_FG     = "#c8402d"
OK_FG      = "#4a9e6e"
NEON_GREEN = "#39ff14"
STAGE_GREEN = "#4ae54a"
BROWSE_BG  = "#252528"
PROG_BG    = "#1a1a1e"

GLOW_PX = 14

# Aura gradient colours (left → right)
AURA = [
    (240, 140, 30),
    (210, 100, 20),
    (80,  50,  160),
    (30,  70,  210),
    (50,  100, 240),
    (70,  80,  220),
    (140, 50,  180),
    (210, 55,  65),
    (180, 40,  35),
]

# stage name → (fraction, label)
STAGE_MAP = {
    "decoding":     (0.05,  "Decoding audio..."),
    "processing":   (0.10,  "Processing blocks..."),
    "flushing":     (0.70,  "Flushing pipeline..."),
    "writing_midi": (0.85,  "Writing MIDI..."),
    "dual_wav":     (0.92,  "Generating dual WAV..."),
    "complete":     (1.00,  "Complete"),
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
LOGO_PATH = os.path.join(PROJECT_DIR, "digital_ear", "paradromics.png")
ICON_PATH = os.path.join(PROJECT_DIR, "digital_ear", "paradromics_icon.png")

# Parameter definitions
PARAMS = [
    ("DC Cutoff (Hz)",            "--dc",           "30.0",   "DC blocker cutoff frequency"),
    ("High-pass (Hz)",            "--hp",           "60.0",   "High-pass filter cutoff"),
    ("Low-pass (Hz)",             "--lp",           "4000.0", "Low-pass filter cutoff"),
    ("Min F0 (Hz)",               "--fmin",         "80.0",   "Min pitch search frequency"),
    ("Max F0 (Hz)",               "--fmax",         "1000.0", "Max pitch search frequency"),
    ("Confidence Threshold",      "--conf-th",      "7.0",    "Pitch confidence threshold"),
    ("RMS Threshold",             "--rms-th",       "0.003",  "RMS energy for voicing"),
    ("Min Note Duration (sec)",   "--min-note-sec", "0.15",   "Discard short notes"),
]

PARAMS_BOTTOM = [
    ("FFT Size",                  "--nfft",         "2048",   "FFT analysis frame length"),
    ("Hop Size",                  "--hop",          "512",    "Samples between frames"),
    ("Sample Rate (Hz)",          "--sr",           "44100",  "Target decode sample rate"),
    ("Block Size",                "--block",        "2048",   "Max block size in samples"),
]

FLAG_PARAMS = [
    ("Polyphonic Mode",  "--poly",  "Multi-voice extraction"),
]

FILE_PARAMS = [
    ("Dump Frames CSV",  "--dump-frames", "Auto-saves per-frame CSV"),
    ("Dual WAV Output",  "--dual",        "Auto-saves stereo WAV"),
]

# GM instruments (name, program number)
GM_INSTRUMENTS = [
    ("Acoustic Grand Piano", 0),
    ("Electric Piano 1", 4),
    ("Harpsichord", 6),
    ("Celesta", 8),
    ("Glockenspiel", 9),
    ("Music Box", 10),
    ("Vibraphone", 11),
    ("Marimba", 12),
    ("Xylophone", 13),
    ("Nylon Guitar", 24),
    ("Steel Guitar", 25),
    ("Jazz Guitar", 26),
    ("Clean Guitar", 27),
    ("Muted Guitar", 28),
    ("Overdriven Guitar", 29),
    ("Distortion Guitar", 30),
    ("Acoustic Bass", 32),
    ("Fingered Bass", 33),
    ("Violin", 40),
    ("Viola", 41),
    ("Cello", 42),
    ("Strings Ensemble", 48),
    ("Synth Strings", 50),
    ("Choir Aahs", 52),
    ("Trumpet", 56),
    ("French Horn", 60),
    ("Soprano Sax", 64),
    ("Alto Sax", 65),
    ("Oboe", 68),
    ("English Horn", 69),
    ("Bassoon", 70),
    ("Clarinet", 71),
    ("Piccolo", 72),
    ("Flute", 73),
    ("Recorder", 74),
    ("Pan Flute", 75),
    ("Ocarina", 79),
    ("Square Lead", 80),
    ("Sawtooth Lead", 81),
    ("Warm Pad", 89),
    ("Polysynth", 90),
]


def _lerp_rgb(colors, t):
    """Lerp through a list of RGB tuples, t in [0,1]."""
    t = max(0.0, min(1.0, t))
    n = len(colors) - 1
    idx = t * n
    lo = int(idx)
    hi = min(lo + 1, n)
    f = idx - lo
    return (
        colors[lo][0] + (colors[hi][0] - colors[lo][0]) * f,
        colors[lo][1] + (colors[hi][1] - colors[lo][1]) * f,
        colors[lo][2] + (colors[hi][2] - colors[lo][2]) * f,
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("The Digital Ear — Paradromics")
        self.configure(bg=BG)
        self.minsize(560, 640)
        self.geometry("660x804")

        self.running = False
        self.process = None
        self.param_vars = {}
        self.flag_vars = {}
        self.file_vars = {}
        self.melody_prog_var = None
        self.bg_prog_var = None
        self._cfg_canvas = None
        self._icon_img = None
        self._logo_img = None
        self._gradient_img = None
        self._aura_after_id = None
        self._aura_size = (0, 0)

        self._build_ui()

    def _build_ui(self):
        self._font       = tkfont.Font(family="Helvetica Neue", size=15)
        self._font_sm    = tkfont.Font(family="Helvetica Neue", size=14)
        self._font_bold  = tkfont.Font(family="Helvetica Neue", size=15, weight="bold")
        self._font_title = tkfont.Font(family="Helvetica Neue", size=19, weight="bold")
        self._font_brand = tkfont.Font(family="PT Mono", size=14, weight="bold")
        self._font_mono  = tkfont.Font(family="Menlo", size=12)
        self._font_gen   = tkfont.Font(family="Helvetica Neue", size=16, weight="bold")
        self._font_cfg   = tkfont.Font(family="Helvetica Neue", size=13)
        self._font_italic = tkfont.Font(family="Helvetica Neue", size=15, slant="italic")

        # background gradient canvas
        self._bg_canvas = tk.Canvas(self, highlightthickness=0, bg=BG)
        self._bg_canvas.pack(fill="both", expand=True)
        self._bg_canvas.bind("<Configure>", self._schedule_aura)

        # content sits inset so gradient border is visible
        content = tk.Frame(self._bg_canvas, bg=BG)
        content.pack(fill="both", expand=True,
                     padx=GLOW_PX, pady=(0, GLOW_PX))
        self._content = content

        # header
        hdr = tk.Frame(content, bg=BG)
        hdr.pack(fill="x", padx=14, pady=(16, 0))

        if os.path.isfile(ICON_PATH):
            try:
                self._icon_img = tk.PhotoImage(file=ICON_PATH)
                self.iconphoto(True, self._icon_img)
            except tk.TclError:
                self._icon_img = None

        if os.path.isfile(LOGO_PATH):
            try:
                raw = tk.PhotoImage(file=LOGO_PATH)
                self._logo_img = raw.subsample(35)
                tk.Label(hdr, image=self._logo_img, bg=BG).pack(
                    side="left", padx=(0, 14))
            except tk.TclError:
                self._logo_img = None

        tk.Label(hdr, text="P  A  R  A  D  R  O  M  I  C  S",
                 font=self._font_brand, fg=FG_DIM, bg=BG).pack(
                     side="left", anchor="w")

        title_frame = tk.Frame(content, bg=BG)
        title_frame.pack(fill="x", padx=14, pady=(14, 2))
        tk.Label(title_frame, text="The Digital Ear", font=self._font_title,
                 fg=FG, bg=BG).pack(anchor="w")
        tk.Label(title_frame, text="Streaming audio-to-MIDI melody extraction",
                 font=self._font_sm, fg=FG_DIM, bg=BG).pack(anchor="w")

        tk.Frame(content, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(14, 0))

        file_frame = tk.Frame(content, bg=BG, padx=14)
        file_frame.pack(fill="x", pady=(12, 0))

        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self._file_row(file_frame, "Input", self.in_var, self._browse_input, 0)
        self._file_row(file_frame, "Output", self.out_var, self._browse_output, 1,
                       placeholder=" auto from input name")

        self.config_visible = False
        self.config_frame = None

        cfg_row = tk.Frame(content, bg=BG, padx=14)
        cfg_row.pack(fill="x", pady=(10, 0))

        self.cfg_btn = tk.Label(
            cfg_row, text="▸  Advanced Config", font=self._font_sm, fg=FG_DIM,
            bg=BG, cursor="hand2",
        )
        self.cfg_btn.pack(side="left")
        self.cfg_btn.bind("<Button-1>", self._toggle_config)

        self.save_config_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            cfg_row, text="Save Config", variable=self.save_config_var,
            font=self._font_cfg, fg=FG_MUTED, bg=BG,
            selectcolor=BG_FIELD, activebackground=BG, activeforeground=FG,
        ).pack(side="right")

        self.debug_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            cfg_row, text="Debug", variable=self.debug_var,
            font=self._font_cfg, fg=FG_MUTED, bg=BG,
            selectcolor=BG_FIELD, activebackground=BG, activeforeground=FG,
        ).pack(side="right", padx=(0, 8))

        self.export_wav_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            cfg_row, text="Export .wav", variable=self.export_wav_var,
            font=self._font_cfg, fg=FG_MUTED, bg=BG,
            selectcolor=BG_FIELD, activebackground=BG, activeforeground=FG,
        ).pack(side="right", padx=(0, 8))

        self.config_container = tk.Frame(content, bg=BG)

        gen_row = tk.Frame(content, bg=BG, padx=14)
        gen_row.pack(fill="x", pady=(12, 0))

        self.gen_btn = tk.Button(
            gen_row, text="GENERATE", font=self._font_gen,
            bg=GEN_BG, fg=GEN_FG, activebackground=GEN_HV, activeforeground=GEN_FG,
            relief="ridge", bd=2, pady=8, cursor="hand2",
            command=self._on_generate,
        )
        self.gen_btn.pack(fill="x")

        tk.Frame(content, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(12, 0))

        log_hdr = tk.Frame(content, bg=BG, padx=14)
        log_hdr.pack(fill="x", pady=(8, 4))
        tk.Label(log_hdr, text="Output", font=self._font_sm, fg=FG_DIM,
                 bg=BG).pack(anchor="w")

        log_outer = tk.Frame(content, bg=LOG_BG, relief="groove", bd=2)
        log_outer.pack(fill="both", expand=True, padx=14, pady=(0, 4))

        self.log_text = tk.Text(
            log_outer, bg=LOG_BG, fg=LOG_FG, font=self._font_mono,
            relief="flat", borderwidth=0, wrap="word",
            insertbackground=FG, selectbackground="#2a2a2e",
            state="disabled", padx=10, pady=8,
            height=10,
        )
        scrollbar = tk.Scrollbar(log_outer, command=self.log_text.yview,
                                 bg=BG_CARD, troughcolor=LOG_BG,
                                 activebackground=BORDER, relief="flat", width=8)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        self.log_text.tag_configure("stat", foreground=STAT_FG)
        self.log_text.tag_configure("cmd", foreground=CMD_FG)
        self.log_text.tag_configure("err", foreground=ERR_FG)
        self.log_text.tag_configure("ok", foreground=OK_FG)

        # progress bar
        prog_frame = tk.Frame(content, bg=BG, padx=14)
        prog_frame.pack(fill="x", pady=(4, 8))

        self._prog_canvas = tk.Canvas(
            prog_frame, height=14, bg=PROG_BG,
            highlightthickness=0, relief="groove", bd=1,
        )
        self._prog_canvas.pack(fill="x")

        self._stage_label = tk.Label(
            prog_frame, text="", font=self._font_cfg,
            fg=STAGE_GREEN, bg=BG, anchor="w",
        )
        self._stage_label.pack(fill="x", pady=(2, 0))

    # --- gradient aura ---

    def _schedule_aura(self, event=None):
        if self._aura_after_id is not None:
            self.after_cancel(self._aura_after_id)
        self._aura_after_id = self.after(100, self._draw_aura)

    def _draw_aura(self):
        c = self._bg_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return
        if (w, h) == self._aura_size and self._gradient_img is not None:
            return
        self._aura_size = (w, h)

        scale = 4
        sw = (w + scale - 1) // scale
        sh = (h + scale - 1) // scale

        h_palette = []
        for x in range(sw):
            hx = x / max(sw - 1, 1)
            h_palette.append(_lerp_rgb(AURA, hx))

        start_row = int(sh * 0.15)
        bg_row = "{" + (" " + BG) * sw + "}"
        rows = []

        for y in range(sh):
            if y < start_row:
                rows.append(bg_row)
            else:
                t = (y - start_row) / max(sh - start_row - 1, 1)
                a = t ** 1.3
                pixels = []
                for r, g, b in h_palette:
                    pixels.append(
                        f"#{int(r * a):02x}{int(g * a):02x}{int(b * a):02x}"
                    )
                rows.append("{" + " ".join(pixels) + "}")

        img = tk.PhotoImage(width=sw, height=sh)
        img.put(" ".join(rows))
        self._gradient_img = img.zoom(scale)

        c.delete("aura")
        c.create_image(0, 0, anchor="nw", image=self._gradient_img, tags="aura")

    def _file_row(self, parent, label, var, browse_cmd, row, placeholder=""):
        tk.Label(parent, text=label, font=self._font_sm, fg=FG_DIM,
                 bg=BG, width=6, anchor="w").grid(row=row, column=0,
                                                   sticky="w", pady=4)
        entry = tk.Entry(
            parent, textvariable=var, font=self._font,
            bg=ENTRY_BG, fg=ENTRY_FG, relief="groove", bd=2,
            insertbackground=FG, selectbackground=BORDER,
        )
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=4, ipady=3)

        if placeholder:
            def _on_focus_in(e, ent=entry, var=var, ph=placeholder):
                if ent.cget("fg") == FG_DIM:
                    var.set("")
                    ent.configure(fg=ENTRY_FG, font=self._font)
            def _on_focus_out(e, ent=entry, var=var, ph=placeholder):
                if not var.get().strip():
                    var.set(ph)
                    ent.configure(fg=FG_DIM, font=self._font_italic)
            def _on_var_change(*a, ent=entry, var=var, ph=placeholder):
                val = var.get()
                if val and val != ph:
                    ent.configure(fg=ENTRY_FG, font=self._font)
            var.set(placeholder)
            entry.configure(fg=FG_DIM, font=self._font_italic)
            entry.bind("<FocusIn>", _on_focus_in)
            entry.bind("<FocusOut>", _on_focus_out)
            var.trace_add("write", _on_var_change)

        btn = tk.Button(
            parent, text="Browse", font=self._font_sm,
            bg=BROWSE_BG, fg=FG_MUTED, activebackground=BORDER,
            activeforeground=FG, relief="groove", bd=1, padx=10, pady=2,
            cursor="hand2", command=browse_cmd,
        )
        btn.grid(row=row, column=2, pady=4)
        parent.columnconfigure(1, weight=1)

    # Config panel

    def _build_config_panel(self):
        f = self.config_container
        f.configure(padx=14, pady=6)

        # scrollable wrapper
        scroll_frame = tk.Frame(f, bg=BG_CARD,
                                highlightbackground=BORDER, highlightthickness=1)
        scroll_frame.pack(fill="x")

        canvas = tk.Canvas(scroll_frame, bg=BG_CARD, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview,
                                 bg="#333338", troughcolor="#111114",
                                 activebackground="#4a4a52",
                                 relief="flat", width=14)
        canvas.configure(yscrollcommand=scrollbar.set)
        self._cfg_canvas = canvas

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG_CARD)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Cap height at 400px
            req_h = inner.winfo_reqheight()
            canvas.configure(height=min(req_h, 420))

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # mousewheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120 or (-1 if event.delta < 0 else 1)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        scroll_frame.bind("<Enter>", _bind_mousewheel)
        scroll_frame.bind("<Leave>", _unbind_mousewheel)

        pad = tk.Frame(inner, bg=BG_CARD, padx=12, pady=10)
        pad.pack(fill="x")

        row = 0

        hdr_row = tk.Frame(pad, bg=BG_CARD)
        hdr_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        tk.Label(hdr_row, text="Parameter Overrides", font=self._font_sm,
                 fg=FG_DIM, bg=BG_CARD).pack(side="left")
        tk.Button(
            hdr_row, text="Reset", font=self._font_cfg,
            bg=BROWSE_BG, fg=FG_MUTED, activebackground=BORDER,
            activeforeground=FG, relief="groove", bd=1, padx=8, pady=0,
            cursor="hand2", command=self._reset_config,
        ).pack(side="right")
        poly_var = tk.BooleanVar(value=False)
        self.flag_vars["--poly"] = poly_var
        tk.Checkbutton(
            hdr_row, text="Polyphonic Mode", variable=poly_var, font=self._font_cfg,
            fg=FG_MUTED, bg=BG_CARD, selectcolor=BG_FIELD,
            activebackground=BG_CARD, activeforeground=FG, anchor="w",
        ).pack(side="right", padx=(0, 12))
        row += 1

        def _add_param_row(parent, r, label, flag, default, tooltip):
            enabled = tk.BooleanVar(value=False)
            value = tk.StringVar(value=default)
            self.param_vars[flag] = (enabled, value)

            cb = tk.Checkbutton(
                parent, text=label, variable=enabled, font=self._font_cfg,
                fg=FG_MUTED, bg=BG_CARD, selectcolor=BG_FIELD,
                activebackground=BG_CARD, activeforeground=FG, anchor="w",
            )
            cb.grid(row=r, column=0, sticky="w", padx=(0, 6))

            entry = tk.Entry(
                parent, textvariable=value, font=self._font_cfg, width=10,
                bg=ENTRY_BG, fg=ENTRY_FG, relief="groove", bd=2,
                insertbackground=FG, selectbackground=BORDER,
                disabledbackground=BG_CARD, disabledforeground=FG_DIM,
                state="disabled",
            )
            entry.grid(row=r, column=1, sticky="w", padx=2, pady=1)

            tk.Label(parent, text=tooltip, font=self._font_cfg, fg=FG_DIM,
                     bg=BG_CARD, anchor="w").grid(row=r, column=2,
                                                   sticky="w", padx=(8, 0))

            def _toggle(entry=entry, var=enabled):
                entry.configure(state="normal" if var.get() else "disabled")
            enabled.trace_add("write", lambda *a, fn=_toggle: fn())

        for label, flag, default, tooltip in PARAMS:
            _add_param_row(pad, row, label, flag, default, tooltip)
            row += 1

        tk.Frame(pad, bg=BORDER, height=1).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        # instrument dropdowns
        gm_names = [name for name, _prog in GM_INSTRUMENTS]

        tk.Label(pad, text="Melody Instrument", font=self._font_cfg,
                 fg=FG_MUTED, bg=BG_CARD, anchor="w").grid(
                     row=row, column=0, sticky="w", padx=(0, 6))
        self.melody_prog_var = tk.StringVar(value="Acoustic Grand Piano")
        mel_menu = tk.OptionMenu(pad, self.melody_prog_var, *gm_names)
        mel_menu.configure(
            font=self._font_cfg, bg=BG_FIELD, fg=ENTRY_FG,
            activebackground=BORDER, activeforeground=FG,
            highlightthickness=0, relief="groove", bd=1,
        )
        mel_menu["menu"].configure(
            bg=BG_FIELD, fg=ENTRY_FG, activebackground=ACCENT,
            activeforeground=FG, font=self._font_cfg,
        )
        mel_menu.grid(row=row, column=1, columnspan=2, sticky="ew", padx=2, pady=1)
        row += 1

        tk.Label(pad, text="Background Instrument", font=self._font_cfg,
                 fg=FG_MUTED, bg=BG_CARD, anchor="w").grid(
                     row=row, column=0, sticky="w", padx=(0, 6))
        self.bg_prog_var = tk.StringVar(value="Steel Guitar")
        bg_menu = tk.OptionMenu(pad, self.bg_prog_var, *gm_names)
        bg_menu.configure(
            font=self._font_cfg, bg=BG_FIELD, fg=ENTRY_FG,
            activebackground=BORDER, activeforeground=FG,
            highlightthickness=0, relief="groove", bd=1,
        )
        bg_menu["menu"].configure(
            bg=BG_FIELD, fg=ENTRY_FG, activebackground=ACCENT,
            activeforeground=FG, font=self._font_cfg,
        )
        bg_menu.grid(row=row, column=1, columnspan=2, sticky="ew", padx=2, pady=1)
        row += 1

        tk.Frame(pad, bg=BORDER, height=1).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1
        for label, flag, tooltip in FILE_PARAMS:
            enabled = tk.BooleanVar(value=False)
            self.file_vars[flag] = (enabled, tk.StringVar(value=""))

            cb = tk.Checkbutton(
                pad, text=label, variable=enabled, font=self._font_cfg,
                fg=FG_MUTED, bg=BG_CARD, selectcolor=BG_FIELD,
                activebackground=BG_CARD, activeforeground=FG, anchor="w",
            )
            cb.grid(row=row, column=0, sticky="w")
            tk.Label(pad, text=tooltip, font=self._font_cfg, fg=FG_DIM,
                     bg=BG_CARD, anchor="w").grid(row=row, column=1,
                                                   columnspan=2, sticky="w",
                                                   padx=(8, 0))
            row += 1

        tk.Frame(pad, bg=BORDER, height=1).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1
        for label, flag, default, tooltip in PARAMS_BOTTOM:
            _add_param_row(pad, row, label, flag, default, tooltip)
            row += 1

        pad.columnconfigure(2, weight=1)
        self.config_frame = inner

    def _toggle_config(self, event=None):
        if self.config_frame is None:
            self._build_config_panel()

        self.config_visible = not self.config_visible
        if self.config_visible:
            self.cfg_btn.configure(text="▾  Advanced Config")
            self.config_container.pack(fill="x", after=self.cfg_btn.master)
            self.config_container.update_idletasks()
            if self._cfg_canvas:
                self._cfg_canvas.configure(scrollregion=self._cfg_canvas.bbox("all"))
        else:
            self.cfg_btn.configure(text="▸  Advanced Config")
            self.config_container.pack_forget()

    def _reset_config(self):
        """Reset config to defaults."""
        defaults = {flag: default for _, flag, default, _ in PARAMS + PARAMS_BOTTOM}

        for flag, (enabled, value) in self.param_vars.items():
            enabled.set(False)
            if flag in defaults:
                value.set(defaults[flag])

        for flag, var in self.flag_vars.items():
            var.set(False)

        for flag, (enabled, _path) in self.file_vars.items():
            enabled.set(False)

        self.debug_var.set(True)
        self.export_wav_var.set(False)

        if self.melody_prog_var:
            self.melody_prog_var.set("Acoustic Grand Piano")
        if self.bg_prog_var:
            self.bg_prog_var.set("Steel Guitar")

    def _save_config_readable(self):
        """Append config snapshot to _CONFIGS.txt."""
        config_path = os.path.join(OUTPUTS_DIR, "_CONFIGS.txt")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # just the filenames
        in_path = self.in_var.get().strip()
        in_name = os.path.basename(in_path) if in_path else "(none)"
        raw_out = self.out_var.get().strip()
        if not raw_out or raw_out == " auto from input name":
            out_name = os.path.splitext(in_name)[0] if in_name != "(none)" else "output"
        else:
            out_name = raw_out

        lines = []
        lines.append(f"{'─' * 50}")
        lines.append(f"  {ts}")
        lines.append(f"  Input:   {in_name}")
        lines.append(f"  Output:  {out_name}.mid")
        lines.append(f"")

        # only include checked overrides
        _all_params = PARAMS + PARAMS_BOTTOM
        label_map = {flag: label for label, flag, _d, _t in _all_params}
        defaults  = {flag: default for _l, flag, default, _t in _all_params}
        overrides = []
        for flag, (enabled, value) in self.param_vars.items():
            if enabled.get():
                label = label_map.get(flag, flag)
                val   = value.get()
                overrides.append(f"    {label:<26s} {val}")

        if overrides:
            lines.append(f"  Parameter Overrides:")
            lines.extend(overrides)
        else:
            lines.append(f"  Parameter Overrides:  (all defaults)")

        flags_on = []
        for label, flag, _tip in FLAG_PARAMS:
            if self.flag_vars.get(flag, tk.BooleanVar()).get():
                flags_on.append(label)
        if self.debug_var.get():
            flags_on.append("Debug")
        if self.export_wav_var.get():
            flags_on.append("Export .wav")
        for label, flag, _tip in FILE_PARAMS:
            ev = self.file_vars.get(flag)
            if ev and ev[0].get():
                flags_on.append(label)

        lines.append(f"  Flags:   {', '.join(flags_on) if flags_on else '(none)'}")

        poly_on = self.flag_vars.get("--poly", tk.BooleanVar()).get()
        mel_name = self.melody_prog_var.get() if self.melody_prog_var else "Acoustic Grand Piano"
        lines.append(f"  Melody Instrument:  {mel_name}")
        if poly_on:
            bg_name = self.bg_prog_var.get() if self.bg_prog_var else "Steel Guitar"
            lines.append(f"  Background Instrument:  {bg_name}")
        lines.append(f"")

        with open(config_path, "a", encoding="utf-8") as cf:
            cf.write("\n".join(lines) + "\n")

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select Input Audio",
            filetypes=[("Audio", "*.m4a *.wav *.mp3 *.flac *.ogg"), ("All", "*.*")],
        )
        if path:
            self.in_var.set(path)
            # auto-fill output name from input
            cur_out = self.out_var.get()
            if not cur_out.strip() or cur_out == " auto from input name":
                base = os.path.splitext(os.path.basename(path))[0]
                self.out_var.set(base)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension="",
            filetypes=[("All", "*.*")],
        )
        if path:
            # strip known extensions
            base = path
            for ext in (".mid", ".midi", ".wav", ".csv"):
                if base.lower().endswith(ext):
                    base = base[:-len(ext)]
                    break
            self.out_var.set(base)

    def _build_args(self):
        args = [sys.executable, "main.py"]
        in_path = self.in_var.get()
        args += ["--in", in_path]

        raw = self.out_var.get().strip()
        if not raw or raw == " auto from input name":
            base = os.path.splitext(os.path.basename(in_path))[0] or "output"
        else:
            base = raw
        args += ["--out", os.path.join("outputs", base + ".mid")]

        for flag, (enabled, value) in self.param_vars.items():
            if enabled.get():
                args += [flag, value.get()]

        if self.debug_var.get():
            args.append("--debug")

        for flag, var in self.flag_vars.items():
            if var.get():
                args.append(flag)

        if self.export_wav_var.get():
            args += ["--wav", os.path.join("outputs", base + ".wav")]

        for flag, (enabled, _path) in self.file_vars.items():
            if enabled.get():
                if flag == "--dump-frames":
                    args += [flag, os.path.join("outputs", base + "_frames.csv")]
                elif flag == "--dual":
                    args += [flag, os.path.join("outputs", base + "_dual.wav")]

        gm_lookup = {name: prog for name, prog in GM_INSTRUMENTS}
        if self.melody_prog_var:
            mel_prog = gm_lookup.get(self.melody_prog_var.get(), 0)
            args += ["--melody-prog", str(mel_prog)]
        if self.bg_prog_var:
            bg_prog = gm_lookup.get(self.bg_prog_var.get(), 26)
            args += ["--bg-prog", str(bg_prog)]

        return args

    def _on_generate(self):
        if self.running:
            return

        in_path = self.in_var.get().strip()

        if not in_path:
            self._log("ERROR: No input file selected.\n", tag="err")
            return
        if not os.path.isfile(in_path):
            self._log(f"ERROR: Input file not found: {in_path}\n", tag="err")
            return

        os.makedirs(OUTPUTS_DIR, exist_ok=True)

        self.running = True
        self.gen_btn.configure(text="RUNNING...", bg=GEN_DIS, fg=FG_DIM,
                               state="disabled")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._reset_progress()

        cmd = self._build_args()

        if self.save_config_var.get():
            self._save_config_readable()

        self._log(f"$ {' '.join(cmd)}\n\n", tag="cmd")

        thread = threading.Thread(target=self._run_subprocess, args=(cmd,),
                                  daemon=True)
        thread.start()

    def _run_subprocess(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=PROJECT_DIR,
            )
            for line in self.process.stdout:
                self.after(0, self._append_line, line)
            self.process.wait()
            rc = self.process.returncode
            if rc == 0:
                self.after(0, self._log, "\nDone.\n", "ok")
            else:
                self.after(0, self._log,
                           f"\nProcess exited with code {rc}\n", "err")
        except Exception as exc:
            self.after(0, self._log, f"\nERROR: {exc}\n", "err")
        finally:
            self.process = None
            self.after(0, self._reset_button)

    _ANSI_RE = re.compile(r"\033\[(\d+)m")
    _ANSI_TAG = {"32": "ok", "31": "err", "0": None}

    def _append_line(self, line):
        # stage lines go to progress bar, not log
        stripped = line.strip()
        if stripped.startswith("STAGE="):
            stage = stripped.split("=", 1)[1]
            self._update_progress(stage)
            return
        if "\033[" in line:
            self._log_ansi(line)
            return
        tag = "stat" if "=" in line and not line.startswith(" ") else None
        self._log(line, tag)

    def _log_ansi(self, line):
        """Parse ANSI codes and insert with matching tags."""
        self.log_text.configure(state="normal")
        parts = self._ANSI_RE.split(line)
        cur_tag = None
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part:
                    if cur_tag:
                        self.log_text.insert("end", part, cur_tag)
                    else:
                        self.log_text.insert("end", part)
            else:
                cur_tag = self._ANSI_TAG.get(part)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log(self, text, tag=None):
        self.log_text.configure(state="normal")
        if tag:
            self.log_text.insert("end", text, tag)
        else:
            self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _reset_button(self):
        self.running = False
        self.gen_btn.configure(text="GENERATE", bg=GEN_BG, fg=GEN_FG,
                               state="normal")

    def _update_progress(self, stage_name):
        if stage_name not in STAGE_MAP:
            return
        fraction, label = STAGE_MAP[stage_name]
        self._set_progress(fraction, label)

    def _set_progress(self, fraction, text=""):
        c = self._prog_canvas
        c.delete("bar")
        w = c.winfo_width()
        h = c.winfo_height()
        if w > 0 and fraction > 0:
            c.create_rectangle(
                0, 0, int(w * fraction), h,
                fill=NEON_GREEN, outline="", tags="bar",
            )
        self._stage_label.configure(text=text)

    def _reset_progress(self):
        self._prog_canvas.delete("bar")
        self._stage_label.configure(text="")


if __name__ == "__main__":
    app = App()
    app.mainloop()
