from util_np import np, r2c
import matplotlib.pyplot as plt


def spectrogram(x, eps= 1e-8):
    """plots spectrogram."""
    assert 2 == x.ndim
    if not np.iscomplexobj(x): x = r2c(x)
    plt.pcolormesh(np.log(np.abs(x.T) + eps))
