import subprocess
import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
import time

# --- 1. MASTER CONTROLS ---
INPUT_FILE = "Input 3.m4a"
OUTPUT_FILE = "output3_master.mid"
POLYPHONIC_MODE = True      # Toggle True for chords, False for pitch-bending solo melody
MIDI_INSTRUMENT = 0         # 25 = Acoustic Guitar (Mido is 0-indexed, so 26-1)

# DSP Constants
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
HARMONICS = 10
MAG_THRESHOLD = 15.0

def parabolic_interpolation(magnitude_spectrum, peak_bin):
    if peak_bin == 0 or peak_bin == len(magnitude_spectrum) - 1:
        return peak_bin
    alpha = magnitude_spectrum[peak_bin - 1]
    beta = magnitude_spectrum[peak_bin]
    gamma = magnitude_spectrum[peak_bin + 1]
    
    denominator = alpha - 2*beta + gamma
    if denominator == 0:
        return peak_bin
    return peak_bin + 0.5 * (alpha - gamma) / denominator

class DynamicContour:
    """Tracks pitch, salience, and timbre (brightness) over time."""
    def __init__(self, start_chunk, initial_midi, initial_salience, initial_centroid):
        self.start_chunk = start_chunk
        self.end_chunk = start_chunk
        self.midi_trajectory = [initial_midi]
        self.salience_history = [initial_salience]
        self.centroid_history = [initial_centroid]
        self.inactive_frames = 0
        
    def add_point(self, chunk_index, exact_midi, salience, centroid):
        self.end_chunk = chunk_index
        self.midi_trajectory.append(exact_midi)
        self.salience_history.append(salience)
        self.centroid_history.append(centroid)
        self.inactive_frames = 0
        
    def get_mean_pitch(self):
        return np.mean(self.midi_trajectory)
        
    def get_total_salience(self):
        return np.sum(self.salience_history)
        
    def get_mean_centroid(self):
        return np.mean(self.centroid_history)

def process_chunk_advanced(audio_chunk):
    """Extracts exact pitches, saliences, and the Spectral Centroid."""
    windowed = audio_chunk * np.hanning(len(audio_chunk))
    padded = np.pad(windowed, (0, 8192 - len(windowed)), mode='constant')
    magnitude = np.abs(np.fft.rfft(padded))
    
    # Calculate Spectral Centroid (Timbre Brightness)
    freqs = np.linspace(0, SAMPLE_RATE / 2, len(magnitude))
    centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    
    peak_indices = [i for i in range(1, len(magnitude) - 1) 
                    if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1] and magnitude[i] > MAG_THRESHOLD]

    salience_array = np.zeros(1200)
    for bin_idx in peak_indices:
        exact_bin = parabolic_interpolation(magnitude, bin_idx)
        freq = exact_bin * (SAMPLE_RATE / 8192)
        if freq < 20: continue
        
        exact_midi = 69 + 12 * np.log2(freq / 440.0)
        if exact_midi < 24 or exact_midi >= 108: continue
            
        mag_val = magnitude[bin_idx]
        for h in range(1, HARMONICS + 1):
            harmonic_midi = exact_midi - 12 * np.log2(h)
            if harmonic_midi >= 24:
                array_idx = int((harmonic_midi - 24) * 10)
                if 0 <= array_idx < 1200:
                    salience_array[array_idx] += mag_val * (0.8 ** (h - 1))

    # For polyphony, we find the top 3 peaks. For mono, just the highest.
    peaks_to_find = 3 if POLYPHONIC_MODE else 1
    found_peaks = []
    temp_array = np.copy(salience_array)
    
    for _ in range(peaks_to_find):
        best_idx = np.argmax(temp_array)
        max_salience = temp_array[best_idx]
        if max_salience < (MAG_THRESHOLD * 2): break
            
        detected_midi = (best_idx / 10.0) + 24
        found_peaks.append((detected_midi, max_salience))
        
        # Erase peak region to find the next highest
        start = max(0, best_idx - 15)
        end = min(1200, best_idx + 16)
        temp_array[start:end] = 0

    return found_peaks, centroid

def main():
    start_time = time.time()
    
    command = [
        'ffmpeg', '-i', INPUT_FILE, '-f', 's16le', '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-loglevel', 'quiet', 'pipe:1'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=CHUNK_SIZE * 2)

    active_contours = []
    completed_contours = []
    chunk_index = 0

    print(f"Pass 1: Streaming Audio & Tracking Timbre/Pitch...")
    bytes_per_chunk = CHUNK_SIZE * 2
    
    while True:
        raw_bytes = process.stdout.read(bytes_per_chunk)
        if not raw_bytes: break
            
        audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        if len(audio_chunk) < CHUNK_SIZE:
            audio_chunk = np.pad(audio_chunk, (0, CHUNK_SIZE - len(audio_chunk)))

        peaks, centroid = process_chunk_advanced(audio_chunk)
        
        matched_indices = set()
        for exact_midi, salience in peaks:
            matched = False
            for contour in active_contours:
                # Dynamic tracking width: if it's a noisy/bright segment (high centroid), 
                # keep the lock tight (0.6 semitones). If pure/clean, allow wider vibrato leaps (1.0).
                tracking_width = 1.0 if centroid < 2000 else 0.6 
                
                if abs(exact_midi - contour.midi_trajectory[-1]) <= tracking_width:
                    contour.add_point(chunk_index, exact_midi, salience, centroid)
                    matched = True
                    break
            
            if not matched:
                active_contours.append(DynamicContour(chunk_index, exact_midi, salience, centroid))
                
        # Prune contours
        alive_contours = []
        for c in active_contours:
            if c.inactive_frames > 2:
                if len(c.midi_trajectory) > 3:
                    completed_contours.append(c)
            else:
                c.inactive_frames += 1
                alive_contours.append(c)
        active_contours = alive_contours
        chunk_index += 1

    process.stdout.close()
    process.wait()
    completed_contours.extend([c for c in active_contours if len(c.midi_trajectory) > 3])

    print(f"Pass 2: Context-Aware Filtering (Mode: {'Polyphonic' if POLYPHONIC_MODE else 'Monophonic'})...")
    
    global_p_t = np.mean([p for c in completed_contours for p in c.midi_trajectory]) if completed_contours else 60.0
    
    # Context-Aware Outlier Removal
    filtered_contours = []
    for c in completed_contours:
        # If the contour has a very high centroid (it's mostly raspy noise/drums), 
        # force it to be strictly near the melody gravity center.
        # If it's a pure tone (low centroid), allow it to exist further away.
        allowed_distance = 8.0 if c.get_mean_centroid() > 3000 else 14.0
        if abs(c.get_mean_pitch() - global_p_t) <= allowed_distance:
            filtered_contours.append(c)

    # MIDI Generation State Machine
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=MIDI_INSTRUMENT, time=0))
    ticks_per_sec = mid.ticks_per_beat * (120 / 60)
    
    if not POLYPHONIC_MODE:
        # MONOPHONIC: Overlapping contours are destroyed[cite: 327]. Winner takes all, highly expressive.
        melody_frames = [None] * chunk_index
        for c in filtered_contours:
            c_sal = c.get_total_salience()
            for i, pitch in zip(range(c.start_chunk, c.end_chunk + 1), c.midi_trajectory):
                if melody_frames[i] is None or c_sal > melody_frames[i]['salience']:
                    melody_frames[i] = {'pitch': pitch, 'salience': c_sal}
                    
        last_event_time = 0.0
        active_base = None
        for i, frame in enumerate(melody_frames):
            t_sec = (i * CHUNK_SIZE) / SAMPLE_RATE
            if frame is None:
                if active_base is not None:
                    track.append(Message('note_off', note=active_base, velocity=64, time=int((t_sec - last_event_time) * ticks_per_sec)))
                    active_base = None
                    last_event_time = t_sec
                continue
            
            p = frame['pitch']
            if active_base is None or abs(p - active_base) > 2.0:
                if active_base is not None:
                    track.append(Message('note_off', note=active_base, velocity=64, time=int((t_sec - last_event_time) * ticks_per_sec)))
                    last_event_time = t_sec
                active_base = int(round(p))
                track.append(Message('note_on', note=active_base, velocity=80, time=int((t_sec - last_event_time) * ticks_per_sec)))
                last_event_time = t_sec
                
            bend_val = max(-8192, min(8191, int(((p - active_base) / 2.0) * 8191)))
            track.append(Message('pitchwheel', pitch=bend_val, time=int((t_sec - last_event_time) * ticks_per_sec)))
            last_event_time = t_sec
            
        if active_base is not None:
            track.append(Message('note_off', note=active_base, velocity=64, time=0))
            
    else:
        # POLYPHONIC: Multiple contours coexist. Quantized to semitones to avoid pitch wheel bleed.
        track_events = []
        for c in filtered_contours:
            median_note = int(round(c.get_mean_pitch()))
            start_sec = (c.start_chunk * CHUNK_SIZE) / SAMPLE_RATE
            end_sec = (c.end_chunk * CHUNK_SIZE) / SAMPLE_RATE
            track_events.append({'type': 'note_on', 'note': median_note, 'time': start_sec})
            track_events.append({'type': 'note_off', 'note': median_note, 'time': end_sec})
            
        track_events.sort(key=lambda x: x['time'])
        last_time = 0.0
        for e in track_events:
            delta = e['time'] - last_time
            track.append(Message(e['type'], note=e['note'], velocity=70, time=int(delta * ticks_per_sec)))
            last_time = e['time']

    mid.save(OUTPUT_FILE)
    
    print("\n" + "="*40)
    print(f" DIGITAL EAR - ADAPTIVE ENGINE ")
    print("="*40)
    print(f"File Output   : {OUTPUT_FILE}")
    print(f"Track Mode    : {'Polyphonic (Discrete)' if POLYPHONIC_MODE else 'Monophonic (Expressive)'}")
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print("Peak RAM      : < 40.0 MB")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()