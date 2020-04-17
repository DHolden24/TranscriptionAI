from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
import imageio
from tempfile import mktemp
import numpy as np
from os import remove

def generate_spectrogram(filename, duration=None):
    mp3_audio = AudioSegment.from_file("Music/MP3/" + filename + ".mp3", format="mp3")
    wname = mktemp('.wav')  # use temporary file
    mp3_audio = mp3_audio.set_channels(1)
    mp3_audio.export(wname, format="wav")  # convert to wav
    FS, data = wavfile.read(wname)  # read wav file

    if duration is None:
        duration = mp3_audio.duration_seconds * 100

    fig = plt.figure(figsize=(duration / 30, 30))
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    plt.specgram(data, Fs=30000, NFFT=1024, noverlap=0, cmap=plt.cm.gray)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0, dpi=50)
    plt.close(fig)

    img = imageio.imread("temp.png")
    remove("temp.png")

    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    return img
