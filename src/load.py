from os.path import join
from scipy.io import wavfile
from scipy.signal import stft
import numpy as np


def load(path, name):
    fs, wav = wavfile.read(join(path, "wavs", name + ".wav"))
    ii = np.iinfo(wav.dtype)
    wav = wav.astype(np.float) / max(abs(ii.min), abs(ii.max))
    f, t, s = stft(wav, fs)
    return s.T


def load_meta(path):
    names = []
    texts = []
    with open(join(path, "metadata.csv")) as file:
        for line in file:
            name, _, text = line.split("|")
            names.append(name)
            texts.append(text)
    return names, texts


def normalize(text, uncase= True, translate= str.maketrans({
        # inconsistant
        "’": "'"
        , "“": '"', "”": '"'
        , "[": "(", "]": ")"
        # rare
        , "â": "a" , "à": "a"
        , "é": "e" , "ê": "e" , "è": "e"
        # just to make it ascii
        , "ü": "ue"})):
    if uncase: text = text.lower()
    return text.translate(translate)
