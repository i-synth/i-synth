#!/usr/bin/env python3


from functools import partial
from os.path import getsize
from util import comp, PointedIndex, encode
from util_io import path, load_meta, load
from util_np import np, jagged_array


names, texts = load_meta()

# take the smallest 1/10 data
sizes = np.array(list(map(comp(getsize, path), names)))
sel = np.split(np.argsort(sizes), 10)[0]
names = names[sel]
texts = texts[sel]

chars = {char for text in texts for char in text}
chars.remove("\n")
chars.remove(" ")
index = PointedIndex(" \n" + "".join(sorted(chars)))
texts = list(map(partial(encode, index), texts))
texts = jagged_array(
    texts
    , fill= index("\n")
    , shape= (len(texts), max(map(len, texts)))
    , dtype= np.uint8)

grams = list(map(comp(load, path), names))
sizes = np.array(list(map(len, grams)))
grams = jagged_array(
    grams
    , fill= complex('(nan+nanj)')
    , shape= (len(grams), max(map(len, grams)), 128)
    , dtype= np.complex64)

np.save("trial/data/index", index.vec)
np.save("trial/data/texts", texts)
np.save("trial/data/grams", grams)
np.save("trial/data/names", names)
