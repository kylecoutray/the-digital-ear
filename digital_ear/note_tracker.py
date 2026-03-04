# Note tracker: converts a stream of per-frame f0 estimates into discrete
# MIDI note events with onset/offset times.
#
# Pipeline:
#   1. Median smoothing (removes single-frame pitch glitches)
#   2. f0 → MIDI note number (with rounding)
#   3. Run-length encoding: consecutive frames with the same note become one event
#   4. Minimum-duration gating: discard notes shorter than a threshold

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math


def hz_to_midi(hz: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    return int(round(69.0 + 12.0 * math.log2(hz / 440.0)))


def midi_to_hz(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


@dataclass
class NoteEvent:
    """A single MIDI note with timing."""
    note: int           # MIDI note number (0-127)
    start_sec: float    # onset time in seconds
    end_sec: float      # offset time in seconds
    velocity: int = 80  # MIDI velocity


@dataclass
class NoteTracker:
    """
    Streaming note tracker that accumulates f0 observations and emits
    NoteEvent objects.

    Parameters
    ----------
    hop_sec : float
        Duration of one hop in seconds (hop_samples / sample_rate).
    median_window : int
        Size of the running median filter (odd number recommended).
    min_note_sec : float
        Minimum note duration in seconds; shorter notes are discarded.
    """

    hop_sec: float
    median_window: int = 9
    min_note_sec: float = 0.12
    merge_gap_sec: float = 0.08  # merge same-pitch notes separated by gaps shorter than this
    outlier_semitones: int = 14  # remove notes >this many semitones from both neighbors
    latency_sec: float = 0.0    # processing chain latency compensation (shifts notes earlier)

    # internal state
    _f0_buf: deque = field(init=False, repr=False)
    _smoothed: list = field(init=False, repr=False)
    _raw_midi: list = field(init=False, repr=False)
    _frame_idx: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._f0_buf = deque(maxlen=self.median_window)
        self._smoothed = []  # list of (midi_note | None, frame_idx)
        self._raw_midi = []  # raw MIDI notes (pre-median) for onset sharpening
        self._frame_idx = 0

    def push(self, f0_hz: float | None) -> None:
        """Feed one frame's f0 estimate (None = unvoiced)."""
        # Store raw MIDI note before median smoothing (for onset sharpening)
        if f0_hz is not None and f0_hz > 0:
            raw_note = hz_to_midi(f0_hz)
            raw_note = max(0, min(127, raw_note))
        else:
            raw_note = None
        self._raw_midi.append(raw_note)

        self._f0_buf.append(f0_hz)
        self._frame_idx += 1

        # Don't emit until the buffer is full (latency = median_window // 2 frames)
        if len(self._f0_buf) < self.median_window:
            return

        # Median of the non-None values in the window
        vals = [v for v in self._f0_buf if v is not None]
        if len(vals) >= (self.median_window // 2 + 1):
            # Majority voiced → take median pitch
            median_hz = sorted(vals)[len(vals) // 2]
            midi_note = hz_to_midi(median_hz)
            midi_note = max(0, min(127, midi_note))
        else:
            midi_note = None

        # The smoothed value corresponds to the centre of the window
        centre_idx = self._frame_idx - (self.median_window // 2) - 1
        self._smoothed.append((midi_note, centre_idx))

    def flush(self) -> list[NoteEvent]:
        """
        Flush remaining buffered frames and return all detected note events.
        Call once after all audio has been processed.
        """
        # Drain remaining samples in the median buffer
        pad_count = self.median_window // 2
        for _ in range(pad_count):
            self.push(None)

        self._octave_correct()
        events = self._run_length_encode()
        events = self._sharpen_boundaries(events)
        events = self._merge_fragments(events)
        events = self._remove_outliers(events)
        return events

    @staticmethod
    def _pitch_matches(raw_note: int, target_note: int) -> bool:
        """Check if a raw MIDI note matches target, allowing octave correction."""
        diff = abs(raw_note - target_note)
        return diff <= 1 or diff == 12 or diff == 19

    def _sharpen_boundaries(self, events: list[NoteEvent]) -> list[NoteEvent]:
        """Refine onset/offset times using raw (pre-median) pitch data.

        The median filter stabilises pitch but blurs note boundaries by up to
        median_window//2 frames (~81 ms).  For each note event we scan the raw
        MIDI values around its boundaries to find the true transition point.
        """
        if not events or not self._raw_midi:
            return events

        half_w = self.median_window // 2
        lat = self.latency_sec
        n_raw = len(self._raw_midi)

        sharpened: list[NoteEvent] = []
        for evt in events:
            # Convert times back to frame indices
            onset_frame = round((evt.start_sec + lat) / self.hop_sec)
            offset_frame = round((evt.end_sec + lat) / self.hop_sec)

            # Sharpen onset: find earliest matching raw frame
            search_lo = max(0, onset_frame - half_w)
            search_hi = min(n_raw, onset_frame + half_w + 1)
            new_onset = onset_frame
            for j in range(search_lo, search_hi):
                raw = self._raw_midi[j]
                if raw is not None and self._pitch_matches(raw, evt.note):
                    new_onset = j
                    break

            # Sharpen offset: find latest matching raw frame
            search_lo2 = max(0, offset_frame - half_w - 1)
            search_hi2 = min(n_raw, offset_frame + half_w)
            new_offset = offset_frame
            for j in range(search_hi2 - 1, search_lo2 - 1, -1):
                raw = self._raw_midi[j]
                if raw is not None and self._pitch_matches(raw, evt.note):
                    new_offset = j + 1
                    break

            start_sec = max(0.0, new_onset * self.hop_sec - lat)
            end_sec = max(start_sec, new_offset * self.hop_sec - lat)

            if end_sec - start_sec >= self.min_note_sec:
                sharpened.append(NoteEvent(
                    note=evt.note, start_sec=start_sec, end_sec=end_sec,
                    velocity=evt.velocity,
                ))

        # Fix overlaps created by boundary adjustment
        for i in range(len(sharpened) - 1):
            if sharpened[i].end_sec > sharpened[i + 1].start_sec:
                mid = (sharpened[i].end_sec + sharpened[i + 1].start_sec) / 2.0
                sharpened[i] = NoteEvent(
                    note=sharpened[i].note,
                    start_sec=sharpened[i].start_sec,
                    end_sec=mid,
                    velocity=sharpened[i].velocity,
                )
                sharpened[i + 1] = NoteEvent(
                    note=sharpened[i + 1].note,
                    start_sec=mid,
                    end_sec=sharpened[i + 1].end_sec,
                    velocity=sharpened[i + 1].velocity,
                )

        return sharpened

    def _merge_fragments(self, events: list[NoteEvent]) -> list[NoteEvent]:
        """Merge consecutive events of the same pitch separated by small gaps."""
        if len(events) < 2:
            return events

        merged: list[NoteEvent] = [events[0]]
        for evt in events[1:]:
            prev = merged[-1]
            gap = evt.start_sec - prev.end_sec
            if evt.note == prev.note and gap <= self.merge_gap_sec:
                # Extend the previous note
                merged[-1] = NoteEvent(
                    note=prev.note,
                    start_sec=prev.start_sec,
                    end_sec=evt.end_sec,
                    velocity=prev.velocity,
                )
            else:
                merged.append(evt)
        return merged

    def _octave_correct(self) -> None:
        """Fix octave/harmonic jumps in the smoothed frame sequence.

        For each voiced frame, compute a local pitch centre from a ±25-frame
        neighbourhood.  If the frame's note is ~12 or ~19 semitones above the
        centre, snap it down by 12 or 19 semitones (it was likely a harmonic
        tracking error).  Similarly snap up if it's an octave below.
        """
        n = len(self._smoothed)
        if n == 0:
            return

        radius = 25  # ~290 ms look-around at hop=512/44100

        for i in range(n):
            note_i, idx_i = self._smoothed[i]
            if note_i is None:
                continue

            # Gather local voiced notes
            local_notes: list[int] = []
            lo = max(0, i - radius)
            hi = min(n, i + radius + 1)
            for j in range(lo, hi):
                nj = self._smoothed[j][0]
                if nj is not None:
                    local_notes.append(nj)

            if len(local_notes) < 5:
                continue

            local_notes.sort()
            median_note = local_notes[len(local_notes) // 2]

            diff = note_i - median_note
            # Snap down if ~1 octave (12 st) or ~octave+fifth (19 st) above
            if 11 <= diff <= 13:
                self._smoothed[i] = (note_i - 12, idx_i)
            elif 18 <= diff <= 20:
                self._smoothed[i] = (note_i - 19, idx_i)
            # Snap up if ~1 octave below
            elif -13 <= diff <= -11:
                self._smoothed[i] = (note_i + 12, idx_i)

    def _remove_outliers(self, events: list[NoteEvent]) -> list[NoteEvent]:
        """Remove notes that are pitch outliers — far from both neighbors."""
        if len(events) < 3:
            return events

        keep: list[NoteEvent] = [events[0]]  # always keep first
        for i in range(1, len(events) - 1):
            prev_dist = abs(events[i].note - events[i - 1].note)
            next_dist = abs(events[i].note - events[i + 1].note)
            # Only remove if far from BOTH neighbors (isolated outlier)
            if prev_dist > self.outlier_semitones and next_dist > self.outlier_semitones:
                continue
            keep.append(events[i])
        keep.append(events[-1])  # always keep last
        return keep

    def _run_length_encode(self) -> list[NoteEvent]:
        """Convert the smoothed note sequence into NoteEvent list.

        Applies latency_sec compensation to shift all note times earlier,
        correcting for processing chain delays (frame windowing, HPSS onset
        softening, etc.).
        """
        events: list[NoteEvent] = []
        if not self._smoothed:
            return events

        lat = self.latency_sec  # shorthand

        cur_note = self._smoothed[0][0]
        cur_start = self._smoothed[0][1]

        for midi_note, idx in self._smoothed[1:]:
            if midi_note != cur_note:
                # Emit previous note (if it was voiced)
                if cur_note is not None:
                    start_sec = max(0.0, cur_start * self.hop_sec - lat)
                    end_sec = max(start_sec, idx * self.hop_sec - lat)
                    duration = end_sec - start_sec
                    if duration >= self.min_note_sec:
                        events.append(NoteEvent(
                            note=cur_note,
                            start_sec=start_sec,
                            end_sec=end_sec,
                        ))
                cur_note = midi_note
                cur_start = idx

        # Final note
        if cur_note is not None:
            last_idx = self._smoothed[-1][1] + 1
            start_sec = max(0.0, cur_start * self.hop_sec - lat)
            end_sec = max(start_sec, last_idx * self.hop_sec - lat)
            duration = end_sec - start_sec
            if duration >= self.min_note_sec:
                events.append(NoteEvent(
                    note=cur_note,
                    start_sec=start_sec,
                    end_sec=end_sec,
                ))

        return events
