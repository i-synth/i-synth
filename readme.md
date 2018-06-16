# model

characters to complex spectrogram frames

R^{s,c} -> R^{t,2f}

- s: number of characters
- t: number of frames
- c: dimension of characters
- f: dimension of spectrograms, aka frequency bins
- d: dimension of model

## encoder

R^{s,c} -> R^{s,d}

## decoder

R^{s,d} -> R^{t,d}

## termination

R^{d} -> Bool

- minimize sigmoid xent

## frame prediction

R^{d} -> R^{2f}

- minimize square error

# src

## main

- `data.py` data preparation
- `transformer_train.py` transformer training

## core

- `transformer.py` transformer model

## util

- `util.py` general
- `util_io.py` for loading, saving, transforming data
- `util_np.py` for numpy
- `util_tf.py` for tensorflow
- `util_plt.py` for matplotlib

# todo

## data

- efficient data loading
- fourier transform parameters
- non-linear spectrogram

## model

- complex algebra
- inspect transformer and optimize attention
- positional encoding for frames
- convolution with pooling to reduce input
