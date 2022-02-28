import numpy as np

P_b = np.array([1, 1])
P_e = np.array([1, 1])

theta = np.pi / 2.0
C_eb = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]
])

C_be = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# pure rotation
print("P_b = C_eb * P_e = ", C_eb @ P_e)
print("P_e = C_be * P_b = ", np.matmul(C_be, P_b))
print("P_e = C_be * C_eb * P_e = ", np.matmul(C_be, C_eb) @ P_e)

O_be = np.array([2, 0])
O_eb = np.array([0, 2])
print("P_e = C_be * P_b + O_be = ", np.matmul(C_be, P_b) + O_be)
print("P_b = C_eb * P_e + O_eb = ", C_eb @ P_e + O_eb)
