import numpy as np
import matplotlib.pyplot as plt

from kinematic_bicycle_model_circle import BicycleCG

sample_time = 0.01
time_end = 30

model_cg = BicycleCG()
model_cg.reset()

t_data = np.arange(0, time_end, sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
v_data = np.zeros_like(t_data)
w_data = np.zeros_like(t_data)

R = 8
max_delta = np.arctan(model_cg.L / R)
print(max_delta)
v_data[:] = 2 * np.pi * R * 2 / time_end

w_data[0: 20] = 1.225
w_data[375 - 20: 375] = -1.225

w_data[375: 375 + 20] = -1.225
w_data[1875 - 20: 1875] = 1.225
w_data[1875: 1875 + 20] = 1.225

for i in range(t_data.shape[0]):
    x_data[i] = model_cg.xc
    y_data[i] = model_cg.yc

    model_cg.step(v_data[i], w_data[i])

plt.axis('square')
plt.xlim(-15, 25)
plt.ylim(-5, 20)
plt.plot(x_data, y_data, label='Square Path')
plt.legend()
plt.show()
