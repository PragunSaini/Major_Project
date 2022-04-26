import pickle
import time
import math
import numpy as np
from collections import defaultdict


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def timeit(method):
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print(f"{method.__name__} {((end - start) * 100):.2f} ms")
        return result
    return timed
