from collections import Counter
from util_io import path, load, save
from util_plt import plt, spectrogram
import numpy as np


def crusher(sample_rate= 2**14, window_size= 2**11, threshold= 1000):
    height = window_size // 2
    bin_size = sample_rate // window_size
    t = threshold // bin_size - 1 # the bin at threshold
    f = (np.arange(height) + 1) * bin_size # ceil frequency for each bin
    h = f[t:] # high bins to crush
    hmel = np.floor(np.log1p(h / threshold) * threshold / np.log(2) / bin_size).astype(np.int)
    # corresponding bins on mel scale
    hbin = Counter(hmel) # the mel bin -> the number of linear bins crushed into
    m = np.zeros((hmel[-1], height))
    for i in range(t): m[i,i] = 1
    for i, b in enumerate(hmel[:-1], t):
        m[b,i] = 1 / hbin[b]
    return m, np.linalg.pinv(m)


param = dict(fs= 22050, nperseg= 2048, noverlap= 1024)
crush, crush_inv = crusher(param['fs'], param['nperseg'])

s = load(path("LJ001-0001"), **param)

s_crushed = s @ crush.T
s2 = s_crushed @ crush_inv.T

plt.subplot(121)
spectrogram(s)
plt.subplot(122)
spectrogram(s2)
plt.show()

save('tmp/original.wav', s, **param)
save('tmp/restored.wav', s2, **param)
