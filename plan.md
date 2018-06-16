# data

- code for loading and converting data
- normalize data
- stft with all default

# model

R^{c,s} -> R^{2f,t}

- c: dimension of char embedding
- f: number of frequency bins (2 * 129)
- d: number of feature maps

## encoder

R^{c,s} -> R^{d,s}

## decoder

R^{d,s} -> R^{d,t}

## terminate: dense

- sigmoid cross entropy

## output: dense

R^{d,t} -> R^{2f,t}

- square/absolute error
