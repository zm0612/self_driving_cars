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
v_data[:] = 2 * np.pi * R * 2 / time_end

w_data[0:0 + 187] = 0.42
w_data[187:187 * 2] = -0.42
w_data[187 * 2:187 * 3] = -0.42
w_data[187 * 3:187 * 4] = 0.42
w_data[187 * 4:187 * 5] = -0.42
w_data[187 * 5:187 * 6] = -0.42

for i in range(t_data.shape[0]):
    x_data[i] = model_cg.xc
    y_data[i] = model_cg.yc

    model_cg.step(v_data[i], w_data[i])

plt.axis('square')
plt.xlim(-40, 40)
plt.ylim(-10, 70)
plt.plot(x_data, y_data, label='Square Path')
plt.legend()
plt.show()
