import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time

class Param_Tracker:
    def __init__(self):
        self.tot = 0
        self.num = 0

    def reset(self):
        self.tot = 0
        self.num = 0

    def __call__(self, val, n=1):
        self.tot += val
        self.num += n

    @property
    def avg(self):
        return self.tot / self.num if self.num > 0 else 0

class Time_Tracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.orig = time.time()
        self.rec = 0
        self.num = 0

    def start(self):
        self.orig = time.time()

    def __call__(self, n=1):
        self.num += n
        self.rec += self.elapsed

    @property
    def avg(self):
        return self.rec / self.num if self.num > 0 else self.elapsed
    
    @property
    def elapsed(self):
        return time.time() - self.orig