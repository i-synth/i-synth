from librosa import load, stft, istft
from librosa.display import specshow
from librosa.output import write_wav
from util_io import path
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


wav, sr = load(path('LJ028-0224'), sr= 16000)


def mel(sr, ws):
    b = ws // 2 + 1 # number of linear frequency bins
    k = 1000 * ws / sr # how many bins for 1kHz
    t = int(k) # the bin at threshold 1000
    h = np.log2(np.linspace(1000, sr//2, b - t) / 1000 + 1) * k # higher bins on mel scale
    m = np.zeros((1 + int(np.ceil(h[-1])), b))
    for i in range(t): m[i,i] = 1.0 # identity for bins below 1kHz
    for i, (f, c, r) in enumerate(zip(np.floor(h), np.ceil(h), np.mod(h, 1)), t):
        # how much of each linear bin should be distributed to the mel bins
        m[int(f),i] += 1 - r
        m[int(c),i] += r
    return m


s = [] # linear spectrograms
# mel vs librosa.filters.mel
m, m2 = [], [] # mel transformations
t, t2 = [], [] # mel spectrograms
r, r2 = [], [] # reconstructed linear spectrograms
w, w2 = [], [] # reconstructed waves
for ws in 256, 512, 1024, 2048, 4096:
    s.append(stft(wav, ws))
    m.append(mel(sr, ws))
    t.append(m[-1] @ s[-1])
    r.append(np.linalg.pinv(m[-1]) @ t[-1])
    w.append(istft(r[-1]))
    m2.append(librosa.filters.mel(sr, ws, len(m[-1]), norm= None))
    t2.append(m2[-1] @ s[-1])
    r2.append(np.linalg.pinv(m2[-1]) @ t2[-1])
    w2.append(istft(r2[-1]))

# frequency bins and time steps
[x.shape for x in s]

# target and source of transformation
[x.shape for x in m]

# librosa.filters.mel has lower error
[np.sum(np.abs(x - wav[:len(x)])) for x in w]
[np.sum(np.abs(x - wav[:len(x)])) for x in w2]

# plot spectrograms
n = 1
for xx in zip(t, t2, r, r2):
    for x in xx:
        plt.subplot(len(s), 4, n)
        specshow(np.log(1e-8 + np.abs(x)))
        n += 1
plt.tight_layout(0)
plt.show()

# plot transformations
n = 1
for xx in m, m2:
    for x in xx:
        plt.subplot(2, len(s), n)
        specshow(x, x_axis= 'linear')
        n += 1
plt.tight_layout(0)
plt.show()

# play reconstructions
for x, x2 in zip(w, w2):
    sd.play(x, sr, blocking= True)
    sd.play(x2, sr, blocking= True)
