import os
from MidiParser import generate_note_data
from SpectrogramGeneration import generate_spectrogram
import numpy as np
import pickle

def generate_data():
    directory = os.fsencode("Music/MIDI")

    sets = []

    for file in os.listdir(directory):
        file = os.fsdecode(file).split('.')[0]
        print(file)

        note_data = generate_note_data(file)
        duration = note_data.shape[0]

        spectrogram = generate_spectrogram(file, duration//4)
        orig_spect_width = spectrogram.shape[1]

        for i in range(20 - orig_spect_width % 10):
            spectrogram = np.append(spectrogram, np.zeros((spectrogram.shape[0], 1)), axis=1)

        spect_width = spectrogram.shape[1]

        tics_per_pixel = duration // spect_width
        diff = (duration % spect_width) / spect_width

        cumulative_diff = 0
        curr_note_frame = 0
        i = 0
        while i < orig_spect_width:
            for x, y in sets:
                pass
            for _ in range(tics_per_pixel):
                curr_note_frame += 1

            cumulative_diff += diff
            i += 1

            if cumulative_diff >= 1:
                sets.append((spectrogram[:, i:i+10], note_data[curr_note_frame]))
                curr_note_frame += 1
                cumulative_diff -= 1
        print(len(sets))

    with open('Music/dataset.pkl', 'wb') as f:
        pickle.dump(sets, f)
    print(len(sets))
