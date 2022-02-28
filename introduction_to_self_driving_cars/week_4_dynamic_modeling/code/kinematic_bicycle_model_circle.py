import numpy as np
import matplotlib.pyplot as plt


class BicycleCR:
    def __init__(self):
        self.xr = 0
        self.yr = 0
        self.theta = 0

        self.L = 2
        self.w_max = 1.22
        self.sample_time = 0.01

    def reset(self):
        self.xr = 0
        self.yr = 0
        self.theta = 0

    def step(self, v, delta):
        xf_dot = v * np.cos(self.theta)
        yf_dot = v * np.sin(self.theta)
        theta_dot = v * np.tan(delta) / self.L

        self.xr += xf_dot * self.sample_time
        self.yr += yf_dot * self.sample_time
        self.theta += theta_dot * self.sample_time


class BicycleCF:
    def __init__(self):
        self.xf = 2
        self.yf = 0
        self.theta = 0

        self.L = 2
        self.sample_time = 0.01

    def reset(self):
        self.xf = 0
        self.yf = 0
        self.theta = 0

    def step(self, v, delta):
        xf_dot = v * np.cos(self.theta + delta)
        yf_dot = v * np.sin(self.theta + delta)
        theta_dot = v * np.sin(delta) / self.L

        self.xf += xf_dot * self.sample_time
        self.yf += yf_dot * self.sample_time
        self.theta += theta_dot * self.sample_time


class BicycleCG:
    def __init__(self):
        self.xc = 1.2
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

    def step(self, v, w):
        xc_dot = v * np.cos(self.theta + self.beta)
        yc_dot = v * np.sin(self.theta + self.beta)
        theta_dot = v * np.cos(self.beta) * np.tan(self.delta) / self.L
        delta_dot = w

        self.xc += xc_dot * self.sample_time
        self.yc += yc_dot * self.sample_time
        self.theta += theta_dot * self.sample_time
        self.delta += delta_dot * self.sample_time
        self.beta = np.arctan(self.lr * np.tan(self.delta) / self.L)


def main():
    delta = np.arctan(2 / 10)

    # 圆形轨迹
    model_cg = BicycleCG()
    model_cg.delta = delta

    model_cf = BicycleCF()
    model_cr = BicycleCR()

    sample_time = 0.01
    t_data = np.arange(0, 20, sample_time)

    R_r = model_cr.L / np.tan(delta)
    R_f = model_cf.L / np.sin(delta)
    R_g = model_cg.L / np.tan(delta) / np.cos(np.arctan(model_cg.lr * np.tan(delta) / model_cg.L))

    print(R_f)
    print(R_g)
    print(R_r)

    v_r = 2 * np.pi * R_r / 20
    v_g = 2 * np.pi * R_g / 20
    v_f = 2 * np.pi * R_f / 20

    # 重心
    x_cg_data = np.zeros_like(t_data)
    y_cg_data = np.zeros_like(t_data)

    # 前轮中心
    x_cf_data = np.zeros_like(t_data)
    y_cf_data = np.zeros_like(t_data)

    # 后轮中心
    x_cr_data = np.zeros_like(t_data)
    y_cr_data = np.zeros_like(t_data)

    for i in range(t_data.shape[0]):
        # 以重心为运动中心
        x_cg_data[i] = model_cg.xc
        y_cg_data[i] = model_cg.yc

        # 以后轮中心为运动中心
        x_cr_data[i] = model_cr.xr
        y_cr_data[i] = model_cr.yr

        # 以前轮中心为运动中心
        x_cf_data[i] = model_cf.xf
        y_cf_data[i] = model_cf.yf

        if model_cg.delta < np.arctan(2 / 10):
            model_cg.step(v_g, model_cg.w_max)
        else:
            model_cg.step(v_g, 0)

        model_cr.step(v_r, np.arctan(2 / 10))
        model_cf.step(v_f, np.arctan(2 / 10))

    plt.axis('equal')
    plt.plot(x_cg_data, y_cg_data, label='cg')
    plt.plot(x_cf_data, y_cf_data, label='cf')
    plt.plot(x_cr_data, y_cr_data, label='cr')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
