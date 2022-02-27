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


model = Bicycle()
model.delta = np.arctan(2 / 10)

sample_time = 0.01
t_data = np.arange(0, 20, sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
x_solution = np.zeros_like(t_data)
y_solution = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc

    if model.delta < np.arctan(2 / 10):
        model.step(np.pi, model.w_max)
    else:
        model.step(np.pi, 0)

    # model.step(np.pi, 0)
    # model.beta = 0

plt.axis('equal')
plt.plot(x_data, y_data, label='Learner Model')
plt.legend()
plt.show()
