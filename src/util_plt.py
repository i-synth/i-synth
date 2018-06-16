import matplotlib.pyplot as plt


def spectrogram(x, eps= 1e-8):
    """plots spectrogram."""
    if not np.iscomplexobj(x): x = r2c(x)
    plt.pcolormesh(np.log(np.abs(x.T) + eps))
