import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Vehicle:
    def __init__(self):
        self.a_0 = 400
        self.a_1 = 0.1
        self.a_2 = -0.0002

        self.GR = 0.35
        self.r_e = 0.3
        self.J_e = 10
        self.m = 2000
        self.g = 9.81

        self.c_a = 1.36
        self.c_r1 = 0.01

        self.c = 10000
        self.F_max = 10000

        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

        self.sample_time = 0.01

    def reset(self):
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

    def step(self, throttle, alpha):
        """
        one step integral
        :param throttle: throttle size (0~1)
        :param alpha: road incline
        :return:
        """
        T_e = throttle * (self.a_0 + self.a_1 * self.w_e + self.a_2 * self.w_e * self.w_e)
        F_aero = self.c_a * self.v * self.v
        R_x = self.c_r1 * np.abs(self.v)
        F_g = self.m * self.g * np.sin(alpha)
        F_load = F_aero + R_x + F_g
        self.w_e_dot = (T_e - self.GR * self.r_e * F_load) / self.J_e

        w_w = self.GR * self.w_e
        s = (w_w * self.r_e - self.v) / self.v

        if np.abs(s) < 1.0:
            F_x = self.c * s
        else:
            F_x = self.F_max

        x_dot_dot = (F_x - F_load) / self.m
        self.v += x_dot_dot * self.sample_time
        self.w_e += self.w_e_dot * self.sample_time


def main():
    sample_time = 0.01
    time_end = 100
    model = Vehicle()

    t_data = np.arange(0, time_end, sample_time)
    v_data = np.zeros_like(t_data)

    throttle = 0.2
    alpha = 0.0

    for i in range(t_data.shape[0]):
        v_data[i] = model.v
        model.step(throttle, alpha)

    plt.plot(t_data, v_data)
    plt.show()


if __name__ == '__main__':
    main()
