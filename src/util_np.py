import numpy as np


def vpack(arrays, fill, offset= 0, extra= 0):
    """like `np.vstack` but for `arrays` of different lengths in the first
    axis.  shorter ones will be padded with `fill` at the end.
    additionally `offset` and `extra` number of columns will be padded
    at the beginning and the end.

    """
    if not hasattr(arrays, '__len__'): arrays = list(arrays)
    arr = arrays[0]
    a = np.full((len(arrays), offset + max(map(len, arrays)) + extra) + arr.shape[1:], fill, arr.dtype)
    for row, arr in zip(a, arrays):
        row[offset:offset+len(arr)] = arr
    return a


def permute(n, seed= 0):
    """returns a random permutation of the first `n` natural numbers."""
    np.random.seed(seed)
    i = np.arange(n)
    np.random.shuffle(i)
    return i


def c2r(x, axis= -1):
    """turns a complex array to a real one by concatenating first the real
    parts and then the imaginary parts along `axis`.

    """
    return np.concatenate((x.real, x.imag), axis)


def r2c(x, axis= -1):
    """undoes `c2r`."""
    r, i = np.split(x, 2, axis)
    return r + 1j * i
