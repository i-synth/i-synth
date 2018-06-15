#!/usr/bin/env python3


from functools import partial
from load import load, load_meta, normalize
from os.path import expanduser, join, getsize
from utils import PointedIndex, jagged_array
import numpy as np


path = expanduser("~/data/LJSpeech-1.0")
names, texts = load_meta(path)
names = np.array(names)
texts = np.array(list(map(normalize, texts)))

# take the smallest 1/10 data
sizes = np.array([getsize(join(path, "wavs", name + ".wav")) for name in names])
sel = np.split(np.argsort(sizes), 10)[0]
names = names[sel]
texts = texts[sel]

chars = {char for text in texts for char in text}
chars.remove("\n")
chars.remove(" ")
index = PointedIndex(" \n" + "".join(sorted(chars)))
texts = jagged_array(
    map(partial(map, index), texts)
    , fill= index("\n")
    , shape= (len(texts), max(map(len, texts)))
    , dtype= np.uint8)
texts = np.concatenate((np.zeros_like(texts[:,:1]), texts), -1)

grams = list(map(partial(load, path), names))
grams = jagged_array(
    grams
    , fill= 0.0
    , shape= (len(grams), max(map(len, grams)), 129)
    , dtype= np.complex64)

np.save("trial/data/index", index.vec)
np.save("trial/data/texts", texts)
np.save("trial/data/grams", grams)
np.save("trial/data/names", names)
