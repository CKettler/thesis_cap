import numpy as np

class statistical_tests:

    def __init__(self, x, y, nmc):
        self.x = x
        self.y = y
        self.nmc = nmc


    def exact_mc_perm_test(self):
        n, k = len(self.x), 0
        diff = float(np.abs(np.mean(self.x) - np.mean(self.y)))
        zs = np.concatenate([self.x, self.y])
        for j in range(self.nmc):
            np.random.shuffle(zs)
            k += diff < abs(float(np.mean(zs[:n])) - float(np.mean(zs[n:])))
        return float(k) / float(self.nmc)
