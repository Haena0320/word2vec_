import math
from numpy.linalg import norm
import numpy as np

def sigmoid(x):
    x = np.float64(x)
    return 1.0 / (1.0 + np.exp(-x))

import math
class UnigramTable:
    def __init__(self, frequency, sample_size):
        self.sample_size = sample_size
        power = 0.75
        norm = sum([math.pow(t, power) for t in frequency.values()])

        table_size = sum(frequency.values())
        table = np.zeros(table_size, dtype=np.int32)
        print("making unigram table start !")

        p = 0
        i = 0
        for j, unigram in enumerate(frequency.values()):
            p += float(math.pow(unigram, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table
        print("making unigram table finish ! ")

    def sample(self):
        indice = np.random.randint(low=0, high=len(self.table), size=5)
        return self.table[indice]

    def get_negative_sample(self, target):
        s_list = list()
        while True:
            s = self.sample()
            if target[0] not in s:
                s_list.append(s)
            else:
                continue
            if len(s_list) == len(target):
                break

        return s_list
