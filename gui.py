#!/usr/bin/env python3
"""
The Digital Ear — Lightweight GUI
tkinter only, zero extra dependencies. Designed for Raspberry Pi.
"""
import datetime
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont

# ── Paradromics colour palette ──────────────────────────────────────
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

# Border width for gradient glow
GLOW_PX = 14

# Aura gradient palette — pumped for visibility
AURA = [
    (240, 140, 30),   # warm orange   (left edge)
    (210, 100, 20),   # deep orange
    (80,  50,  160),  # purple
    (30,  70,  210),  # deep blue
    (50,  100, 240),  # bright blue   (centre)
    (70,  80,  220),  # blue
    (140, 50,  180),  # purple-red
    (210, 55,  65),   # crimson
    (180, 40,  35),   # dark red      (right edge)
]

# Progress stage mapping: stage_name → (fraction, label)
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

# ── Parameter definitions ───────────────────────────────────────────
PARAMS = [
    ("Sample Rate (Hz)",          "--sr",           "44100",  "Target decode sample rate"),
    ("Block Size",                "--block",        "2048",   "Max block size in samples"),
    ("DC Cutoff (Hz)",            "--dc",           "30.0",   "DC blocker cutoff frequency"),
    ("High-pass (Hz)",            "--hp",           "60.0",   "High-pass filter cutoff"),
    ("Low-pass (Hz)",             "--lp",           "4000.0", "Low-pass filter cutoff"),
    ("FFT Size",                  "--nfft",         "2048",   "FFT analysis frame length"),
    ("Hop Size",                  "--hop",          "512",    "Samples between frames"),
    ("Min F0 (Hz)",               "--fmin",         "80.0",   "Min pitch search frequency"),
    ("Max F0 (Hz)",               "--fmax",         "1000.0", "Max pitch search frequency"),
    ("Confidence Threshold",      "--conf-th",      "7.0",    "Pitch confidence threshold"),
    ("RMS Threshold",             "--rms-th",       "0.003",  "RMS energy for voicing"),
    ("Min Note Duration (sec)",   "--min-note-sec", "0.15",   "Discard short notes"),
]

FLAG_PARAMS = [
    ("Polyphonic Mode",  "--poly",  "Multi-voice extraction"),
]

FILE_PARAMS = [
    ("Dump Frames CSV",  "--dump-frames", "Auto-saves per-frame CSV"),
    ("Dual WAV Output",  "--dual",        "Auto-saves stereo WAV"),
]


def _lerp_rgb(colors, t):
    """Interpolate through a list of (r,g,b) tuples. t in [0,1]."""
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
        self.geometry("660x820")

        self.running = False
        self.process = None
        self.param_vars = {}
        self.flag_vars = {}
        self.file_vars = {}
        self._icon_img = None
        self._logo_img = None
        self._gradient_img = None
        self._aura_after_id = None
        self._aura_size = (0, 0)

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        self._font       = tkfont.Font(family="Helvetica Neue", size=14)
        self._font_sm    = tkfont.Font(family="Helvetica Neue", size=13)
        self._font_bold  = tkfont.Font(family="Helvetica Neue", size=14, weight="bold")
        self._font_title = tkfont.Font(family="Helvetica Neue", size=18, weight="bold")
        self._font_brand = tkfont.Font(family="PT Mono", size=13)
        self._font_mono  = tkfont.Font(family="Menlo", size=11)
        self._font_gen   = tkfont.Font(family="Helvetica Neue", size=15, weight="bold")
        self._font_cfg   = tkfont.Font(family="Helvetica Neue", size=12)

        # ── Background gradient canvas (full window) ──
        self._bg_canvas = tk.Canvas(self, highlightthickness=0, bg=BG)
        self._bg_canvas.pack(fill="both", expand=True)
        self._bg_canvas.bind("<Configure>", self._schedule_aura)

        # ── Content frame (margins expose the gradient border) ──
        content = tk.Frame(self._bg_canvas, bg=BG)
        content.pack(fill="both", expand=True,
                     padx=GLOW_PX, pady=(0, GLOW_PX))
        self._content = content

        # ── Header: logo + brand ──
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

        # ── Title ──
        title_frame = tk.Frame(content, bg=BG)
        title_frame.pack(fill="x", padx=14, pady=(14, 2))
        tk.Label(title_frame, text="The Digital Ear", font=self._font_title,
                 fg=FG, bg=BG).pack(anchor="w")
        tk.Label(title_frame, text="Streaming audio-to-MIDI melody extraction",
                 font=self._font_sm, fg=FG_DIM, bg=BG).pack(anchor="w")

        tk.Frame(content, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(14, 0))

        # ── File selection ──
        file_frame = tk.Frame(content, bg=BG, padx=14)
        file_frame.pack(fill="x", pady=(12, 0))

        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self._file_row(file_frame, "Input", self.in_var, self._browse_input, 0)
        self._file_row(file_frame, "Output", self.out_var, self._browse_output, 1,
                       placeholder="auto from input name")

        # ── Config toggle + Save Config ──
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

        # ── Generate button ──
        gen_row = tk.Frame(content, bg=BG, padx=14)
        gen_row.pack(fill="x", pady=(12, 0))

        self.gen_btn = tk.Button(
            gen_row, text="GENERATE", font=self._font_gen,
            bg=GEN_BG, fg=GEN_FG, activebackground=GEN_HV, activeforeground=GEN_FG,
            relief="ridge", bd=2, pady=8, cursor="hand2",
            command=self._on_generate,
        )
        self.gen_btn.pack(fill="x")

        # ── Log / output ──
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

        # ── Progress bar + stage label ──
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

    # ── Gradient aura ────────────────────────────────────────────────

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

    # ── File rows ────────────────────────────────────────────────────

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

        # Placeholder text (greyed-out hint when field is empty)
        if placeholder:
            def _on_focus_in(e, ent=entry, var=var, ph=placeholder):
                if ent.cget("fg") == FG_DIM:
                    var.set("")
                    ent.configure(fg=ENTRY_FG)
            def _on_focus_out(e, ent=entry, var=var, ph=placeholder):
                if not var.get().strip():
                    var.set(ph)
                    ent.configure(fg=FG_DIM)
            def _on_var_change(*a, ent=entry, var=var, ph=placeholder):
                val = var.get()
                if val and val != ph:
                    ent.configure(fg=ENTRY_FG)
            # Set initial placeholder
            var.set(placeholder)
            entry.configure(fg=FG_DIM)
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

    # ── Config panel ─────────────────────────────────────────────────

    def _build_config_panel(self):
        f = self.config_container
        f.configure(padx=14, pady=6)

        inner = tk.Frame(f, bg=BG_CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        inner.pack(fill="x")

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
        row += 1

        for label, flag, default, tooltip in PARAMS:
            enabled = tk.BooleanVar(value=False)
            value = tk.StringVar(value=default)
            self.param_vars[flag] = (enabled, value)

            cb = tk.Checkbutton(
                pad, text=label, variable=enabled, font=self._font_cfg,
                fg=FG_MUTED, bg=BG_CARD, selectcolor=BG_FIELD,
                activebackground=BG_CARD, activeforeground=FG, anchor="w",
            )
            cb.grid(row=row, column=0, sticky="w", padx=(0, 6))

            entry = tk.Entry(
                pad, textvariable=value, font=self._font_cfg, width=10,
                bg=ENTRY_BG, fg=ENTRY_FG, relief="groove", bd=2,
                insertbackground=FG, selectbackground=BORDER,
                disabledbackground=BG_CARD, disabledforeground=FG_DIM,
                state="disabled",
            )
            entry.grid(row=row, column=1, sticky="w", padx=2, pady=1)

            tk.Label(pad, text=tooltip, font=self._font_cfg, fg=FG_DIM,
                     bg=BG_CARD, anchor="w").grid(row=row, column=2,
                                                   sticky="w", padx=(8, 0))

            def _toggle(entry=entry, var=enabled):
                entry.configure(state="normal" if var.get() else "disabled")
            enabled.trace_add("write", lambda *a, fn=_toggle: fn())
            row += 1

        tk.Frame(pad, bg=BORDER, height=1).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        for label, flag, tooltip in FLAG_PARAMS:
            var = tk.BooleanVar(value=False)
            self.flag_vars[flag] = var
            cb = tk.Checkbutton(
                pad, text=label, variable=var, font=self._font_cfg,
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

        # File-output toggles (paths auto-generated — no entry fields)
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

        pad.columnconfigure(2, weight=1)
        self.config_frame = inner

    def _toggle_config(self, event=None):
        if self.config_frame is None:
            self._build_config_panel()

        self.config_visible = not self.config_visible
        if self.config_visible:
            self.cfg_btn.configure(text="▾  Advanced Config")
            self.config_container.pack(fill="x", after=self.cfg_btn.master)
        else:
            self.cfg_btn.configure(text="▸  Advanced Config")
            self.config_container.pack_forget()

    def _reset_config(self):
        """Reset all config params to defaults, uncheck all checkboxes."""
        # Build a quick lookup: flag → default value
        defaults = {flag: default for _, flag, default, _ in PARAMS}

        for flag, (enabled, value) in self.param_vars.items():
            enabled.set(False)
            if flag in defaults:
                value.set(defaults[flag])

        for flag, var in self.flag_vars.items():
            var.set(False)

        for flag, (enabled, _path) in self.file_vars.items():
            enabled.set(False)

        # Debug defaults to on in GUI
        self.debug_var.set(True)
        self.export_wav_var.set(False)

    # ── Config snapshot ────────────────────────────────────────────

    def _save_config_readable(self):
        """Append a human-readable config snapshot to _CONFIGS.txt."""
        config_path = os.path.join(OUTPUTS_DIR, "_CONFIGS.txt")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Input / output names (just filenames, not full paths)
        in_path = self.in_var.get().strip()
        in_name = os.path.basename(in_path) if in_path else "(none)"
        raw_out = self.out_var.get().strip()
        if not raw_out or raw_out == "auto from input name":
            out_name = os.path.splitext(in_name)[0] if in_name != "(none)" else "output"
        else:
            out_name = raw_out

        lines = []
        lines.append(f"{'─' * 50}")
        lines.append(f"  {ts}")
        lines.append(f"  Input:   {in_name}")
        lines.append(f"  Output:  {out_name}.mid")
        lines.append(f"")

        # Overridden parameters (only the ones the user actually checked)
        label_map = {flag: label for label, flag, _d, _t in PARAMS}
        defaults  = {flag: default for _l, flag, default, _t in PARAMS}
        overrides = []
        for flag, (enabled, value) in self.param_vars.items():
            if enabled.get():
                label = label_map.get(flag, flag)
                val   = value.get()
                dflt  = defaults.get(flag, "")
                note  = f"  (default {dflt})" if val != dflt else ""
                overrides.append(f"    {label:<26s} {val}{note}")

        if overrides:
            lines.append(f"  Parameter Overrides:")
            lines.extend(overrides)
        else:
            lines.append(f"  Parameter Overrides:  (all defaults)")

        # Flags
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
        lines.append(f"")

        with open(config_path, "a", encoding="utf-8") as cf:
            cf.write("\n".join(lines) + "\n")

    # ── Browse dialogs ───────────────────────────────────────────────

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select Input Audio",
            filetypes=[("Audio", "*.m4a *.wav *.mp3 *.flac *.ogg"), ("All", "*.*")],
        )
        if path:
            self.in_var.set(path)
            # Auto-populate output name from input filename (if user hasn't set one)
            cur_out = self.out_var.get().strip()
            if not cur_out or cur_out == "auto from input name":
                base = os.path.splitext(os.path.basename(path))[0]
                self.out_var.set(base)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension="",
            filetypes=[("All", "*.*")],
        )
        if path:
            # Strip known extensions — user only sees the base name
            base = path
            for ext in (".mid", ".midi", ".wav", ".csv"):
                if base.lower().endswith(ext):
                    base = base[:-len(ext)]
                    break
            self.out_var.set(base)

    # ── Build CLI args ───────────────────────────────────────────────

    def _build_args(self):
        args = [sys.executable, "main.py"]
        in_path = self.in_var.get()
        args += ["--in", in_path]

        # Auto-append extensions, route to outputs/ folder
        raw = self.out_var.get().strip()
        if not raw or raw == "auto from input name":
            # Derive from input filename
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

        # Top-level "Export .wav" — render MIDI as standalone WAV
        if self.export_wav_var.get():
            args += ["--wav", os.path.join("outputs", base + ".wav")]

        for flag, (enabled, _path) in self.file_vars.items():
            if enabled.get():
                if flag == "--dump-frames":
                    args += [flag, os.path.join("outputs", base + "_frames.csv")]
                elif flag == "--dual":
                    args += [flag, os.path.join("outputs", base + "_dual.wav")]

        return args

    # ── Generate ─────────────────────────────────────────────────────

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

        # Ensure outputs/ directory exists
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

        self.running = True
        self.gen_btn.configure(text="RUNNING...", bg=GEN_DIS, fg=FG_DIM,
                               state="disabled")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._reset_progress()

        cmd = self._build_args()

        # Save config to debug file if enabled
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

    # ANSI SGR codes we care about → tkinter tag names
    _ANSI_RE = re.compile(r"\033\[(\d+)m")
    _ANSI_TAG = {"32": "ok", "31": "err", "0": None}

    def _append_line(self, line):
        # Intercept STAGE= lines for progress bar (don't log them)
        stripped = line.strip()
        if stripped.startswith("STAGE="):
            stage = stripped.split("=", 1)[1]
            self._update_progress(stage)
            return
        # Lines with ANSI colour codes → parse and apply tags
        if "\033[" in line:
            self._log_ansi(line)
            return
        tag = "stat" if "=" in line and not line.startswith(" ") else None
        self._log(line, tag)

    def _log_ansi(self, line):
        """Parse ANSI colour codes and insert with matching tkinter tags."""
        self.log_text.configure(state="normal")
        parts = self._ANSI_RE.split(line)
        cur_tag = None
        for i, part in enumerate(parts):
            if i % 2 == 0:          # text fragment
                if part:
                    if cur_tag:
                        self.log_text.insert("end", part, cur_tag)
                    else:
                        self.log_text.insert("end", part)
            else:                    # ANSI code number
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

    # ── Progress bar ─────────────────────────────────────────────────

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
