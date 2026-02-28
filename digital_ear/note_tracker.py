from __future__ import annotations

from dataclasses import dataclass
import math


def f0_to_midi(f0_hz: float) -> float:
    # MIDI note number for frequency
    return 69.0 + 12.0 * math.log2(f0_hz / 440.0)


def quantize_midi(m: float) -> int:
    return int(round(m))


@dataclass
class NoteEvent:
    note: int
    t_on: float
    t_off: float

    @property
    def dur(self) -> float:
        return self.t_off - self.t_on


@dataclass
class NoteTracker:
    """
    Hysteresis-based monophonic note tracker.

    Rules:
      - Start a note only after N consecutive frames agree on candidate note.
      - End a note only after M consecutive unvoiced frames.
      - If voiced switches to a different note, treat as "candidate" until stable.
    """
    n_on: int = 3          # frames required to confirm a note-on
    n_off: int = 3         # frames required to confirm note-off (unvoiced)
    min_dur: float = 0.05  # seconds, discard shorter notes

    # state
    current_note: int | None = None
    t_on: float = 0.0
    last_voiced_t: float = 0.0

    cand_note: int | None = None
    cand_count: int = 0
    unvoiced_count: int = 0

    def update(self, t: float, f0_hz_or_none: float | None) -> list[NoteEvent]:
        events: list[NoteEvent] = []

        if f0_hz_or_none is None:
            self.unvoiced_count += 1

            if self.current_note is not None and self.unvoiced_count >= self.n_off:
                t_off = self.last_voiced_t
                ev = NoteEvent(note=self.current_note, t_on=self.t_on, t_off=t_off)
                if ev.dur >= self.min_dur:
                    events.append(ev)

                # reset to idle
                self.current_note = None
                self.cand_note = None
                self.cand_count = 0

            return events

        # voiced frame
        self.unvoiced_count = 0
        self.last_voiced_t = t

        midi = f0_to_midi(f0_hz_or_none)
        q = quantize_midi(midi)

        if self.current_note is None:
            # idle: build candidate until stable
            if self.cand_note == q:
                self.cand_count += 1
            else:
                self.cand_note = q
                self.cand_count = 1

            if self.cand_count >= self.n_on:
                # note-on
                self.current_note = self.cand_note
                self.t_on = t
                # clear candidate
                self.cand_note = None
                self.cand_count = 0

            return events

        # currently in a note
        if q == self.current_note:
            # stable, clear candidate
            self.cand_note = None
            self.cand_count = 0
            return events

        # different note while voiced: treat as pending change
        if self.cand_note == q:
            self.cand_count += 1
        else:
            self.cand_note = q
            self.cand_count = 1

        if self.cand_count >= self.n_on:
            # end current note at last voiced time (just before change feels reasonable)
            t_off = t
            ev = NoteEvent(note=self.current_note, t_on=self.t_on, t_off=t_off)
            if ev.dur >= self.min_dur:
                events.append(ev)

            # start new note
            self.current_note = self.cand_note
            self.t_on = t
            self.cand_note = None
            self.cand_count = 0

        return events

    def flush(self, t_end: float) -> list[NoteEvent]:
        """
        Call at end of stream to close any active note.
        """
        events: list[NoteEvent] = []
        if self.current_note is not None:
            t_off = max(self.last_voiced_t, t_end)
            ev = NoteEvent(note=self.current_note, t_on=self.t_on, t_off=t_off)
            if ev.dur >= self.min_dur:
                events.append(ev)
        # reset not strictly required here
        return events