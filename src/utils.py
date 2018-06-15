def jagged_array(x, fill, shape, dtype):
    """-> np.ndarray; with jagged `x` trimmed and filled into `shape`."""
    import numpy as np
    a = np.full(shape= shape, dtype= dtype, fill_value= fill)
    i, *shape = shape
    if shape:
        x = np.stack([jagged_array(x, fill, shape, dtype) for x in x])
    else:
        x = np.fromiter(x, dtype= dtype)
    i = min(i, len(x))
    a[:i] = x[:i]
    return a


def permute(n, seed= 0):
    """returns a random permutation of the first `n` natural numbers."""
    import numpy as np
    np.random.seed(0)
    i = np.arange(n)
    np.random.shuffle(i)
    return i


def batch(data, batch_size, shuffle= 1e4, repeat= True, name= "batch"):
    """returns a tensorflow dataset iterator from `data`."""
    import tensorflow as tf
    with tf.variable_scope(name):
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle: ds = ds.shuffle(int(shuffle))
        if repeat:  ds = ds.repeat()
        return ds.batch(batch_size) \
                 .make_one_shot_iterator() \
                 .get_next()


class PointedIndex:
    """takes a vector of unique elements `vec` and a base index `nil`
    within its range, returns a pointed index `idx`, such that:

    `idx[i]` returns the element at index `i`;

    `idx(x)` returns the index of element `x`;

    bijective for all indices and elements within `vec`;

    otherwise `idx(x) => nil` and `idx[i] => vec[nil]`.

    """

    def __init__(self, vec, nil= 0):
        self._nil = nil
        self._i2x = vec
        self._x2i = {x: i for i, x in enumerate(vec)}

    def __repr__(self):
        return "{}(vec= {}, nil= {})".format(
            type(self).__name__
            , repr(self._i2x)
            , repr(self._nil))

    def __getitem__(self, i):
        try:
            return self._i2x[i]
        except IndexError:
            return self._i2x[self._nil]

    def __call__(self, x):
        try:
            return self._x2i[x]
        except KeyError:
            return self._nil

    def __len__(self):
        return len(self._i2x)

    @property
    def vec(self):
        return self._i2x

    @property
    def nil(self):
        return self._nil


class Record(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __bool__(self):
        return bool(self.__dict__)

    def __call__(self, attr):
        return getattr(self, attr)

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, k, d= None):
        return self.__dict__.get(k, d)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__.items())

    def keys(self):
        return self.__dict__.keys()
