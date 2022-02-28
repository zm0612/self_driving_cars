import numpy as np
import matplotlib.pyplot as plt

from kinematic_bicycle_model_circle import BicycleCG

sample_time = 0.01
time_end = 60

model_cg = BicycleCG()
model_cg.reset()
model_cg.sample_time = sample_time
model_cg.xc = 0
model_cg.yc = 0

t_data = np.arange(0, time_end, sample_time)
x_cr_data = np.zeros_like(t_data)
y_cr_data = np.zeros_like(t_data)

v_data = np.zeros_like(t_data)
v_data[:] = 4

w_data = np.zeros_like(t_data)

w_data[670:670 + 100] = 0.753
w_data[670 + 100:670 + 100 * 2] = -0.753
w_data[2210:2210 + 100] = 0.753
w_data[2210 + 100:2210 + 100 * 2] = -0.753
w_data[3670:3670 + 100] = 0.753
w_data[3670 + 100:3670 + 100 * 2] = -0.753
w_data[5220:5220 + 100] = 0.753
w_data[5220 + 100:5220 + 100 * 2] = -0.753

for i in range(t_data.shape[0]):
    x_cr_data[i] = model_cg.xc
    y_cr_data[i] = model_cg.yc

    model_cg.step(v_data[i], w_data[i])

plt.axis('square')
plt.xlim(-40, 40)
plt.ylim(-10, 70)
plt.plot(x_cr_data, y_cr_data, label='Square Path')
plt.legend()
plt.show()
