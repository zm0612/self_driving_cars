import numpy as np
import matplotlib.pyplot as plt


class Bicycle:
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22
        self.sample_time = 0.01

    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
