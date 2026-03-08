# Minimal MIDI Type 0 (single-track) file writer.
# No external dependencies — writes the binary format directly.
#
# Supports: note_on, note_off with delta-time encoding.
# Enough for a monophonic melody.

from __future__ import annotations

import struct
from dataclasses import dataclass


def _var_len(value: int) -> bytes:
    """Encode an integer as a MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("Negative value")
    buf = []
    buf.append(value & 0x7F)
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.reverse()
    return bytes(buf)


@dataclass
class MIDINote:
    """A note event for the MIDI writer."""
    note: int          # 0-127
    start_tick: int    # absolute tick
    end_tick: int      # absolute tick
    velocity: int = 80
    channel: int = 0


def write_midi(
    notes: list[MIDINote],
    ticks_per_beat: int,
    path: str,
    programs: dict[int, int] | None = None,
) -> None:
    """Write a Type 0 MIDI file. Notes must be sorted by start_tick.
    programs: optional dict of channel -> GM program number (inserts at tick 0)."""
    # collect (absolute_tick, event_bytes)
    events: list[tuple[int, bytes]] = []

    # program changes at tick 0
    if programs:
        for ch, prog in sorted(programs.items()):
            events.append((0, bytes([0xC0 | (ch & 0x0F), prog & 0x7F])))

    for n in notes:
        note_val = max(0, min(127, n.note))
        vel = max(1, min(127, n.velocity))
        ch = n.channel & 0x0F

        # note_on
        events.append((n.start_tick, bytes([0x90 | ch, note_val, vel])))
        # note_off
        events.append((n.end_tick, bytes([0x80 | ch, note_val, 0])))

    # sort by tick (note_off before note_on at same tick)
    events.sort(key=lambda e: (e[0], e[1][0] & 0xF0 != 0x80))

    # build track data with delta times
    track_data = bytearray()
    prev_tick = 0
    for tick, evt in events:
        delta = tick - prev_tick
        if delta < 0:
            delta = 0
        track_data.extend(_var_len(delta))
        track_data.extend(evt)
        prev_tick = tick

    # end-of-track
    track_data.extend(_var_len(0))
    track_data.extend(b'\xFF\x2F\x00')

    # tempo: 120 BPM = 500000 us/beat
    tempo_event = bytearray()
    tempo_event.extend(_var_len(0))               # delta = 0
    tempo_event.extend(b'\xFF\x51\x03')            # meta tempo, length 3
    tempo_event.extend(struct.pack('>I', 500000)[1:])  # 3 bytes, big-endian
    track_data = bytes(tempo_event) + bytes(track_data)

    # assemble file
    with open(path, 'wb') as f:
        # MThd header
        f.write(b'MThd')
        f.write(struct.pack('>I', 6))             # header length
        f.write(struct.pack('>HHH', 0, 1, ticks_per_beat))  # format 0, 1 track

        # MTrk
        f.write(b'MTrk')
        f.write(struct.pack('>I', len(track_data)))
        f.write(track_data)


def seconds_to_ticks(sec: float, ticks_per_beat: int, bpm: float = 120.0) -> int:
    """Convert seconds to MIDI ticks at the given tempo."""
    beats_per_sec = bpm / 60.0
    return int(round(sec * beats_per_sec * ticks_per_beat))
