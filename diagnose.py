#!/usr/bin/env python3
"""
Drop this in your the-digital-ear/ folder and run:
  python diagnose.py --in m3.m4a
It shows you exactly what frequencies dominate the song and what
the detector will see, so we can tune fmin/fmax correctly.
"""
import subprocess, sys, argparse
import numpy as np

def decode(path, sr=44100):
    cmd = ["ffmpeg","-v","error","-i",path,"-f","s16le",
           "-ac","1","-ar",str(sr),"pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    x = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)/32768.0
    return x, sr

def analyze(path):
    print(f"Analyzing {path}...")
    x, sr = decode(path)
    dur = len(x)/sr
    print(f"Duration: {dur:.1f}s  |  Peak: {np.abs(x).max():.3f}  |  RMS: {np.sqrt(np.mean(x**2)):.4f}\n")

    n_fft = 2048
    hop   = 512
    win   = np.hamming(n_fft)
    n_harmonics = 4

    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    # --- Global dominant frequencies ---
    mag_full = np.abs(np.fft.rfft(x[:min(len(x), sr*30)] *
                                   np.hamming(min(len(x), sr*30))))
    freqs = np.linspace(0, sr/2, len(mag_full))
    k_lo, k_hi = int(60*len(mag_full)*2/sr), int(2000*len(mag_full)*2/sr)
    band = mag_full[k_lo:k_hi]
    top = np.argsort(band)[-15:][::-1]
    print("Top frequencies in full mix (60-2000 Hz):")
    seen = set()
    for i in top:
        f = freqs[k_lo+i]
        midi = int(round(69 + 12*np.log2(f/440))) if f>0 else 0
        name = note_names[midi%12]+str(midi//12-1)
        if name not in seen:
            print(f"  {f:7.1f} Hz  {name:>5}  mag={band[i]:.0f}")
            seen.add(name)

    # --- Frame-by-frame: what does HPS detect at each moment? ---
    print(f"\nHPS detections per frequency band:")
    bands = [(60,150,"BASS (skip this)"),(150,300,"LOW-MID melody"),
             (300,600,"MID melody"),(600,1200,"HIGH melody")]

    for flo, fhi, label in bands:
        k_min = max(1, int(flo*n_fft/sr))
        k_max = min(n_fft//2, int(fhi*n_fft/sr))
        k_max_hps = min(k_max, (n_fft//2)//(n_harmonics+1))
        note_counts = {}
        for start in range(0, len(x)-n_fft, hop):
            frame = x[start:start+n_fft]
            rms = np.sqrt(np.mean(frame**2))
            if rms < 2e-3: continue
            mag = np.maximum(np.abs(np.fft.rfft(frame*win)), 1e-12)
            lm = np.log(mag)
            hps = lm.copy()
            for h in range(2, n_harmonics+2):
                ds = lm[::h]; l = min(len(hps),len(ds))
                hps[:l] += ds[:l]
            band_h = hps[k_min:k_max_hps+1]
            k_best = k_min + int(np.argmax(band_h))
            f0 = k_best*sr/n_fft
            conf = np.exp(float(band_h.max())-float(np.median(band_h)))
            if conf < 20: continue
            midi = int(round(69+12*np.log2(f0/440))) if f0>0 else 0
            name = note_names[midi%12]+str(midi//12-1)
            note_counts[name] = note_counts.get(name,0)+1
        top_notes = sorted(note_counts.items(), key=lambda x:-x[1])[:6]
        total = sum(note_counts.values())
        notes_str = "  ".join(f"{n}({c})" for n,c in top_notes)
        print(f"  {flo:4}-{fhi:4} Hz  [{label:22}]  {total:4} frames  |  {notes_str}")

    print("\nRECOMMENDATION:")
    print("Set --fmin to the bottom of whichever band has the most melody frames")
    print("Set --fmax to the top of that band")
    print("e.g. if most melody is in 150-600 Hz: --fmin 150 --fmax 650")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="path", required=True)
    args = p.parse_args()
    analyze(args.path)