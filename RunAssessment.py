import pickle, os, mido
from SpectrogramGeneration import generate_spectrogram
import numpy as np

with open('ai.pkl', 'rb') as f:
    cnn = pickle.load(f)


directory = os.fsencode("Music/ToProcess")

for file in os.listdir(directory):
    file = os.fsdecode(file).split('.')[0]
    print("Converting {}".format(file))

    spectrogram = generate_spectrogram(file)
    orig_spect_width = spectrogram.shape[1]

    for i in range(20 - orig_spect_width % 10):
        spectrogram = np.append(spectrogram, np.zeros((spectrogram.shape[0], 1)), axis=1)

    notes = [[0] * 128]

    for i in range(500):
        if i%1000 == 0: print("Done {} of {}".format(i, orig_spect_width))
        spect_slice = spectrogram[:, i:i + 10]
        spect_slice.shape = (1, spect_slice.shape[0], spect_slice.shape[1])
        slice_notes = cnn.predict(spect_slice)
        norm_result = np.zeros(slice_notes.shape)
        for j in range(slice_notes.shape[0]):
            if slice_notes[j] > 0.7:
                norm_result[j] = 1
            else:
                norm_result[j] = 0
        notes.append(norm_result.tolist())

    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.UnknownMetaMessage(type_byte=0x08, data=(84, 104, 101, 32, 69, 110, 100, 32, 105, 115, 32, 71, 111, 111, 100, 0), time=0))
    track.append(mido.UnknownMetaMessage(type_byte=0x0a, data=(73, 122, 122, 121, 68, 105, 122, 122, 121, 76, 105, 0), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    track.append(mido.MetaMessage("key_signature", key='C', time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("program_change", channel=0, program=24, time=0))
    track.append(mido.Message('note_on', channel=0, note=81, velocity=80, time=0))
    track.append(mido.Message("control_change", channel=0, control=7, value=100, time=2))
    track.append(mido.Message("control_change", channel=0, control=10, value=64, time=2))
    track.append(mido.Message("control_change", channel=0, control=91, value=30, time=2))
    track.append(mido.Message("control_change", channel=0, control=93, value=30, time=2))
    prev_time = 0

    for i in range(1, len(notes)):
        prev = notes[i-1]
        curr = notes[i]

        for j in range(len(curr)):
            if prev[j] != curr[j]:
                if curr[j] != 0:
                    track.append(mido.Message('note_on', channel=0, note=j, velocity=80, time=2*(i-1)-prev_time))
                else:
                    track.append(mido.Message('note_on', channel=0, note=j, velocity=0, time=2*(i-1)-prev_time))
                prev_time = 2 * (i - 1)

    midi.save("Music/ToProcess/{}.mid".format(file))

