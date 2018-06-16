from os.path import expanduser, join
from scipy.io import wavfile
from scipy.signal import stft, istft
from util_np import np, r2c


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


def path(name, folder= "~/data/LJSpeech-1.0"):
    """returns the absolute path for LJSpeech audio with `name`."""
    return join(expanduser(folder), "wavs", name + ".wav")


def load_meta(path= "~/data/LJSpeech-1.0", filename= "metadata.csv", sep= "|", normalize= normalize):
    """reads LJSpeech metadata.

    returns 2 str arrays, of filenames and normalized texts.

    """
    names, texts = [], []
    with open(join(expanduser(path), filename)) as file:
        for line in file:
            name, _, text = line.split(sep)
            names.append(name)
            texts.append(normalize(text))
    return np.array(names), np.array(texts)


def load(path, fs= 22050):
    """reads a wav file and returns the complex spectrogram.

    the first axis are the time steps and the second axis the
    frequency bins, with the first frequency bin removed.

    """
    fs2, wav = wavfile.read(path)
    assert fs == fs2
    if np.issubdtype(wav.dtype, np.integer):
        ii = np.iinfo(wav.dtype)
        wav = wav.astype(np.float) / max(abs(ii.min), abs(ii.max))
    f, t, s = stft(wav, fs)
    return s[1:].T


def save(path, x, fs= 22050, dtype= np.int16):
    """undoes `load`."""
    if not np.iscomplexobj(x): x = r2c(x)
    x = x.T
    t, wav = istft(np.concatenate((np.zeros_like(x[:1]), x)))
    if np.issubdtype(dtype, np.integer):
        ii = np.iinfo(dtype)
        wav *= max(abs(ii.min), abs(ii.max))
    wavfile.write(path, fs, wav.astype(dtype))


def plot(x, eps= 1e-8):
    """plots spectrogram."""
    import matplotlib.pyplot as plt
    if not np.iscomplexobj(x): x = r2c(x)
    plt.pcolormesh(np.log(np.abs(x.T) + eps))
