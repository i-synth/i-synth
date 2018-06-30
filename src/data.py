#!/usr/bin/env python3


from os.path import getsize
from util import comp, partial, PointedIndex
from util_io import path, load_meta, load
from util_np import np, vpack


names, texts = load_meta()

chars = {char for text in texts for char in text}
chars.remove("\n")
chars.remove(" ")
index = PointedIndex(" \n" + "".join(sorted(chars)))
texts = vpack(map(comp(partial(np.fromiter, dtype= np.uint8), partial(map, index)), texts), index("\n"))

np.save("trial/data/index", index.vec)
np.save("trial/data/texts", texts)
np.save("trial/data/names", names)

# for name, gram in zip(names, grams):
#     np.save("trial/data/grams/" + name, load(path(name)))
