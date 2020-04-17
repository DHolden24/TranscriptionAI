from mido import MidiFile
import numpy as np

def generate_note_data(filename):
    mid = MidiFile("Music/MIDI/" + filename + ".mid", clip=True)

    track = mid.tracks[0]
    duration = sum([m.time for m in track])

    notes = np.zeros((duration // 2, 128))
    note_on = [0] * 128

    messages = [m for m in track if m.type=="note_on"]

    if len(messages) == 0:
        track = mid.tracks[1]
        messages = [m for m in track if m.type=="note_on"]

    t = 0
    try:
        next_time = messages[0].time
    except IndexError:
        print(mid)
        raise IndexError

    while (t + 2) < duration:
        if len(messages) > 0 and next_time <= t:
            note = messages[0].note
            note_on[note] = 1 if messages[0].velocity != 0 else 0
            messages.pop(0)

            while len(messages) > 0 and messages[0].time < 2:
                note = messages[0].note
                note_on[note] = 1 if messages[0].velocity != 0 else 0
                messages.pop(0)

            if len(messages) > 0:
                next_time = messages[0].time + t

        notes[t // 2] = note_on
        t += 2

    return notes

generate_note_data('10000_Ukuleles')
