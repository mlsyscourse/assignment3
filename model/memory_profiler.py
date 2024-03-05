import numpy as np


class MemoryProfiler(object):
    def __init__(self):
        self.peak_memory = 0
        self.temp_memory = 0

    def start(self):
        self.temp_memory = 0

    def end(self):
        if self.temp_memory > self.peak_memory:
            self.peak_memory = self.temp_memory
        self.temp_memory = 0

    def add_var(self, x: np.ndarray):
        self.temp_memory += x.size * x.itemsize
