from os.path import expanduser, join
from util_np import np, r2c
import librosa


def path(name, folder= "~/data/LJSpeech-1.0"):
    """returns the absolute path for LJSpeech audio with `name`."""
    return join(expanduser(folder), "wavs", name + ".wav")


def normalize(text, padl= " ", padr= "\n", uncase= True, translate= str.maketrans({
        # inconsistent
        "’": "'"
        , "“": '"', "”": '"'
        , "[": "(", "]": ")"
        # rare
        , "â": "a" , "à": "a"
        , "é": "e" , "ê": "e" , "è": "e"
        # just to make it ascii
        , "ü": "ue"})):
    if uncase: text = text.lower()
    text = text.translate(translate)
    if not text.startswith(padl): text = padl + text
    if not text.endswith(padr): text = text + padr
    return text


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


def load(path, rate= 8000, nfft= 512, hop= 256):
    """reads a wav file and returns the complex spectrogram.

    the first axis are the time steps and the second axis the
    frequency bins, with the first frequency bin removed.

    """
    wav, _ = librosa.load(path, rate)
    frame = librosa.stft(wav, nfft, hop)
    return frame[:-1].T


def save(path, x, rate= 8000, hop= 256):
    """undoes `load`."""
    assert 2 == x.ndim
    if np.isrealobj(x): x = r2c(x)
    x = x.T
    x = np.concatenate((x, np.zeros_like(x[:1])))
    wav = librosa.istft(x, hop)
    librosa.output.write_wav(path, wav, rate)
