from __future__ import annotations

import mido


def seconds_to_ticks(t_sec: float, bpm: float, ppq: int) -> int:
    ticks_per_second = (ppq * bpm) / 60.0
    return int(round(t_sec * ticks_per_second))


def write_monophonic_midi(
    note_events,
    out_path: str,
    bpm: float = 120.0,
    ppq: int = 480,
    program: int = 0,
    velocity: int = 80,
) -> None:
    """
    note_events: iterable of objects with .note (int), .t_on (sec), .t_off (sec)
    Writes a single-track monophonic MIDI (we do not enforce monophony here,
    but your tracker already emits non-overlapping notes).
    """
    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.Message("program_change", program=int(program), time=0))

    msgs = []
    for ev in note_events:
        on = seconds_to_ticks(float(ev.t_on), bpm, ppq)
        off = seconds_to_ticks(float(ev.t_off), bpm, ppq)
        if off <= on:
            off = on + 1

        note = int(ev.note)
        if note < 0:
            note = 0
        if note > 127:
            note = 127

        msgs.append((on, mido.Message("note_on", note=note, velocity=int(velocity), time=0)))
        msgs.append((off, mido.Message("note_off", note=note, velocity=0, time=0)))

    msgs.sort(key=lambda x: x[0])

    last = 0
    for tick, msg in msgs:
        msg.time = tick - last
        last = tick
        track.append(msg)

    mid.save(out_path)