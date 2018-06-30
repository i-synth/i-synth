#!/usr/bin/env python3


from os.path import getsize
from util import comp, partial, PointedIndex
from util_io import path, load_meta, load
from util_np import np, vpack


names, texts = load_meta()

# # take the smallest 1/10 data
# sizes = np.array(list(map(comp(getsize, path), names)))
# sel = np.split(np.argsort(sizes), 10)[0]
# names = names[sel]
# texts = texts[sel]

chars = {char for text in texts for char in text}
chars.remove("\n")
chars.remove(" ")
index = PointedIndex(" \n" + "".join(sorted(chars)))
texts = vpack(map(comp(partial(np.fromiter, dtype= np.uint8), partial(map, index)), texts), index("\n"))

# grams = vpack(map(comp(load, path), names), complex('(nan+nanj)'), 1, 1)
# grams[:,0] = 0j
# np.save("trial/data/grams", grams)

np.save("trial/data/index", index.vec)
np.save("trial/data/texts", texts)
np.save("trial/data/names", names)
