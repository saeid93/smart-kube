import numpy as np

class Discrete2MultiDiscrete:
    """convertor of discrete action to multi-discrete action"""
    def __init__(self, m, n):
            self.m = m
            self.n = n
    def number_to_base(self, n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
    def __getitem__(self, i):
            assert i < self.m**self.n, "Index out of range" 
            res = [0 for _ in range(self.n)]
            p = self.number_to_base(i, self.m)
            if len(p) < self.n:
                res[-len(p):] = p
                return np.array(res)
            else:
                return np.array(p)
