# Note tracker: f0 stream -> discrete MIDI note events
#
# Streaming with bounded memory. Median smoothing, octave correction,
# run-length encoding into pending notes, then post-processing
# (sharpen boundaries, merge, outlier removal, reonset splitting).

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
    """Streaming note tracker: f0 estimates -> NoteEvents with bounded memory.
    Notes are emitted incrementally from push()."""

    hop_sec: float
    median_window: int = 9
    min_note_sec: float = 0.12
    merge_gap_sec: float = 0.08
    outlier_semitones: int = 14
    latency_sec: float = 0.0
    reonset_attack: float = 0.0
    reonset_min_gap: float = 0.0
    reonset_min_dur: float = 0.5

    # -- internal state --
    # median filter buffer
    _f0_buf: deque = field(init=False, repr=False)

    # sliding window for octave correction: (midi_note|None, abs_frame_idx)
    _smooth_window: deque = field(init=False, repr=False)
    # how many smoothed frames have been octave-corrected and fed to RLE
    _oct_cursor: int = field(init=False, repr=False)

    # ring buffers for raw MIDI and RMS (sharpen + reonset)
    _raw_midi_buf: deque = field(init=False, repr=False)
    _rms_ring: deque = field(init=False, repr=False)
    _ring_start_idx: int = field(init=False, repr=False)  # absolute frame idx of ring[0]

    # run-length encoder state
    _rle_note: int | None = field(init=False, repr=False)  # current note (or None)
    _rle_start_idx: int = field(init=False, repr=False)     # start frame of current run
    _rle_started: bool = field(init=False, repr=False)

    # pending notes (need neighbors for outlier check)
    _pending: deque = field(init=False, repr=False)
    _last_emitted: NoteEvent | None = field(init=False, repr=False)

    _frame_idx: int = field(init=False, repr=False)
    _flushing: bool = field(init=False, repr=False)

    # constants
    _OCT_RADIUS: int = 25       # octave correction look-around (frames)
    _RING_SIZE: int = 500       # raw_midi + rms ring buffer size
    _PENDING_DEPTH: int = 3     # notes buffered before oldest can be finalized

    def __post_init__(self) -> None:
        self._f0_buf = deque(maxlen=self.median_window)
        self._smooth_window = deque()
        self._oct_cursor = 0
        self._raw_midi_buf = deque(maxlen=self._RING_SIZE)
        self._rms_ring = deque(maxlen=self._RING_SIZE)
        self._ring_start_idx = 0
        self._rle_note = None
        self._rle_start_idx = 0
        self._rle_started = False
        self._pending = deque()
        self._last_emitted = None
        self._frame_idx = 0
        self._flushing = False

    def push(self, f0_hz: float | None, rms_val: float = 0.0) -> list[NoteEvent]:
        """Feed one frame's f0 estimate. Returns 0+ finalized NoteEvents."""

        # store raw MIDI + RMS
        if f0_hz is not None and f0_hz > 0:
            raw_note = hz_to_midi(f0_hz)
            raw_note = max(0, min(127, raw_note))
        else:
            raw_note = None

        # track ring buffer offset
        if len(self._raw_midi_buf) == self._RING_SIZE:
            self._ring_start_idx += 1
        self._raw_midi_buf.append(raw_note)
        self._rms_ring.append(rms_val)

        # median filter
        self._f0_buf.append(f0_hz)
        self._frame_idx += 1

        if len(self._f0_buf) < self.median_window:
            return []

        vals = [v for v in self._f0_buf if v is not None]
        if len(vals) >= (self.median_window // 2 + 1):
            median_hz = sorted(vals)[len(vals) // 2]
            midi_note = hz_to_midi(median_hz)
            midi_note = max(0, min(127, midi_note))
        else:
            midi_note = None

        centre_idx = self._frame_idx - (self.median_window // 2) - 1
        self._smooth_window.append((midi_note, centre_idx))

        # octave-correct frames that have enough trailing context
        emitted = self._process_smoothed_window()
        return emitted

    def flush(self) -> list[NoteEvent]:
        """End of stream: drain remaining frames and pending notes."""
        self._flushing = True
        result: list[NoteEvent] = []

        # drain median buffer
        pad_count = self.median_window // 2
        for _ in range(pad_count):
            result.extend(self.push(None))

        # process remaining smoothed frames (no more context coming)
        while self._oct_cursor < len(self._smooth_window):
            self._octave_correct_frame(self._oct_cursor)
            note, idx = self._smooth_window[self._oct_cursor]
            self._oct_cursor += 1
            self._rle_feed(note, idx)

        # close final RLE run
        if self._rle_started and self._rle_note is not None:
            evt = self._rle_emit(self._smooth_window[-1][1] + 1 if self._smooth_window else self._rle_start_idx)
            if evt is not None:
                self._pending.append(evt)

        # flush pending notes
        result.extend(self._drain_pending(force_all=True))

        self._flushing = False
        return result

    def _process_smoothed_window(self) -> list[NoteEvent]:
        """Process smoothed frames that have enough trailing context for octave correction."""
        emitted: list[NoteEvent] = []

        # frame at position i needs _OCT_RADIUS frames after it
        while self._oct_cursor < len(self._smooth_window) - self._OCT_RADIUS:
            self._octave_correct_frame(self._oct_cursor)
            note, idx = self._smooth_window[self._oct_cursor]
            self._oct_cursor += 1
            rle_events = self._rle_feed(note, idx)
            emitted.extend(rle_events)

            # trim old frames, keep enough for look-back
            trim_target = self._oct_cursor - self._OCT_RADIUS
            if trim_target > 0:
                for _ in range(trim_target):
                    self._smooth_window.popleft()
                self._oct_cursor -= trim_target

        return emitted

    def _octave_correct_frame(self, pos: int) -> None:
        """Snap frame at `pos` down/up an octave if it's far from the local median."""
        note_i, idx_i = self._smooth_window[pos]
        if note_i is None:
            return

        n = len(self._smooth_window)
        lo = max(0, pos - self._OCT_RADIUS)
        hi = min(n, pos + self._OCT_RADIUS + 1)

        local_notes: list[int] = []
        for j in range(lo, hi):
            nj = self._smooth_window[j][0]
            if nj is not None:
                local_notes.append(nj)

        if len(local_notes) < 5:
            return

        local_notes.sort()
        median_note = local_notes[len(local_notes) // 2]

        diff = note_i - median_note
        if 11 <= diff <= 13:
            self._smooth_window[pos] = (note_i - 12, idx_i)
        elif 18 <= diff <= 20:
            self._smooth_window[pos] = (note_i - 19, idx_i)
        elif -13 <= diff <= -11:
            self._smooth_window[pos] = (note_i + 12, idx_i)

    def _rle_feed(self, midi_note: int | None, frame_idx: int) -> list[NoteEvent]:
        """Feed one frame to the RLE. Returns finalized events when a note
        boundary causes the pending queue to drain."""
        emitted: list[NoteEvent] = []

        if not self._rle_started:
            self._rle_note = midi_note
            self._rle_start_idx = frame_idx
            self._rle_started = True
            return emitted

        if midi_note != self._rle_note:
            # note boundary, close previous run
            evt = self._rle_emit(frame_idx)
            if evt is not None:
                self._pending.append(evt)
                # try to finalize old pending notes
                emitted.extend(self._drain_pending(force_all=False))

            self._rle_note = midi_note
            self._rle_start_idx = frame_idx

        return emitted

    def _rle_emit(self, end_frame_idx: int) -> NoteEvent | None:
        """Close current RLE run into a NoteEvent (None if unvoiced/too short)."""
        if self._rle_note is None:
            return None

        lat = self.latency_sec
        start_sec = max(0.0, self._rle_start_idx * self.hop_sec - lat)
        end_sec = max(start_sec, end_frame_idx * self.hop_sec - lat)

        if end_sec - start_sec < self.min_note_sec:
            return None

        return NoteEvent(
            note=self._rle_note,
            start_sec=start_sec,
            end_sec=end_sec,
        )

    def _drain_pending(self, force_all: bool = False) -> list[NoteEvent]:
        """Finalize pending notes. Needs _PENDING_DEPTH notes buffered so
        outlier check can see prev + next. force_all=True drains everything."""
        emitted: list[NoteEvent] = []

        while len(self._pending) > 0:
            if not force_all and len(self._pending) < self._PENDING_DEPTH:
                break

            note = self._pending.popleft()

            # sharpen boundaries
            note = self._sharpen_single(note)
            if note is None:
                continue

            # reonset split
            split_notes = self._reonset_single(note)

            for sn in split_notes:
                # outlier check
                if self._is_outlier(sn):
                    continue

                # merge if same pitch + small gap
                if self._try_merge(sn):
                    continue  # merged into _last_emitted

                self._last_emitted = sn
                emitted.append(sn)

        return emitted

    @staticmethod
    def _pitch_matches(raw_note: int, target_note: int) -> bool:
        """Does raw_note match target (allowing octave correction)?"""
        diff = abs(raw_note - target_note)
        return diff <= 1 or diff == 12 or diff == 19

    def _raw_midi_at(self, abs_idx: int) -> int | None:
        """Get raw MIDI value at absolute frame index, or None if out of range."""
        local = abs_idx - self._ring_start_idx
        if 0 <= local < len(self._raw_midi_buf):
            return self._raw_midi_buf[local]
        return None

    def _rms_at(self, abs_idx: int) -> float:
        """Get RMS value at absolute frame index, or 0 if out of range."""
        local = abs_idx - self._ring_start_idx
        if 0 <= local < len(self._rms_ring):
            return self._rms_ring[local]
        return 0.0

    def _sharpen_single(self, evt: NoteEvent) -> NoteEvent | None:
        """Sharpen onset/offset using raw MIDI data."""
        half_w = self.median_window // 2
        lat = self.latency_sec

        onset_frame = round((evt.start_sec + lat) / self.hop_sec)
        offset_frame = round((evt.end_sec + lat) / self.hop_sec)

        # find earliest matching raw frame
        new_onset = onset_frame
        for j in range(max(0, onset_frame - half_w), onset_frame + half_w + 1):
            raw = self._raw_midi_at(j)
            if raw is not None and self._pitch_matches(raw, evt.note):
                new_onset = j
                break

        # find latest matching raw frame
        new_offset = offset_frame
        for j in range(offset_frame + half_w - 1, max(-1, offset_frame - half_w - 2), -1):
            raw = self._raw_midi_at(j)
            if raw is not None and self._pitch_matches(raw, evt.note):
                new_offset = j + 1
                break

        start_sec = max(0.0, new_onset * self.hop_sec - lat)
        end_sec = max(start_sec, new_offset * self.hop_sec - lat)

        if end_sec - start_sec < self.min_note_sec:
            return None

        # fix overlap with previous note
        if self._last_emitted is not None and start_sec < self._last_emitted.end_sec:
            mid = (start_sec + self._last_emitted.end_sec) / 2.0
            self._last_emitted = NoteEvent(
                note=self._last_emitted.note,
                start_sec=self._last_emitted.start_sec,
                end_sec=mid,
                velocity=self._last_emitted.velocity,
            )
            start_sec = mid

        return NoteEvent(
            note=evt.note,
            start_sec=start_sec,
            end_sec=end_sec,
            velocity=evt.velocity,
        )

    def _is_outlier(self, evt: NoteEvent) -> bool:
        """Is this note a pitch outlier relative to its neighbors?"""
        prev = self._last_emitted
        # peek at next pending note
        nxt = self._pending[0] if self._pending else None

        if prev is None or nxt is None:
            return False  # can't check without both neighbors

        prev_dist = abs(evt.note - prev.note)
        next_dist = abs(evt.note - nxt.note)
        return prev_dist > self.outlier_semitones and next_dist > self.outlier_semitones

    def _try_merge(self, evt: NoteEvent) -> bool:
        """Try to merge evt into _last_emitted. Returns True if merged."""
        if self._last_emitted is None:
            return False

        gap = evt.start_sec - self._last_emitted.end_sec
        if evt.note == self._last_emitted.note and gap <= self.merge_gap_sec:
            self._last_emitted = NoteEvent(
                note=self._last_emitted.note,
                start_sec=self._last_emitted.start_sec,
                end_sec=evt.end_sec,
                velocity=self._last_emitted.velocity,
            )
            return True
        return False

    def _reonset_single(self, evt: NoteEvent) -> list[NoteEvent]:
        """Split a note at RMS attack transients. Returns 1+ notes."""
        if self.reonset_attack <= 0:
            return [evt]

        lat = self.latency_sec
        hop = self.hop_sec
        effective_gap = self.reonset_min_gap if self.reonset_min_gap > 0 else self.min_note_sec
        min_gap_frames = max(1, int(effective_gap / hop))
        rms_alpha = 0.95

        start_frame = int(round((evt.start_sec + lat) / hop))
        end_frame = int(round((evt.end_sec + lat) / hop))

        note_dur = evt.end_sec - evt.start_sec
        if note_dur < self.reonset_min_dur or end_frame - start_frame < 2 * min_gap_frames:
            return [evt]

        # scan for RMS attack transients
        split_frames: list[int] = []
        rms_ema = self._rms_at(start_frame)
        frames_since_last = 0

        for fi in range(start_frame + 1, end_frame):
            frames_since_last += 1
            cur_rms = self._rms_at(fi)

            if (rms_ema > 0
                    and cur_rms > self.reonset_attack * rms_ema
                    and frames_since_last >= min_gap_frames):
                split_frames.append(fi)
                frames_since_last = 0
                rms_ema = cur_rms

            rms_ema = rms_alpha * rms_ema + (1.0 - rms_alpha) * cur_rms

        if not split_frames:
            return [evt]

        # split into sub-notes
        result: list[NoteEvent] = []
        boundaries = [start_frame] + split_frames + [end_frame]
        for i in range(len(boundaries) - 1):
            s = boundaries[i]
            e = boundaries[i + 1]
            s_sec = max(0.0, s * hop - lat)
            e_sec = max(s_sec, e * hop - lat)
            if e_sec - s_sec >= self.min_note_sec:
                result.append(NoteEvent(
                    note=evt.note,
                    start_sec=s_sec,
                    end_sec=e_sec,
                    velocity=evt.velocity,
                ))

        return result if result else [evt]
