import numpy as np
import matplotlib.pyplot as plt


class VehicleCF:
    def __init__(self, xf=0, yf=0, theta=0, v=0):
        self.xf = xf
        self.yf = yf
        self.theta = theta
        self.v = v

        self.L = 2
        self.sample_time = 0.1

    def step(self, v, delta, a):
        xf_dot = v * np.cos(self.theta + delta)
        yf_dot = v * np.sin(self.theta + delta)
        theta_dot = v * np.sin(delta) / self.L

        self.xf += xf_dot * self.sample_time
        self.yf += yf_dot * self.sample_time
        self.theta += theta_dot * self.sample_time
        self.v += a * self.sample_time


def stanley_control(vehicle, t_x, t_y, t_theta, k):
    """
    Stanley Controller
    refer to: 《Autonomous automobile trajectory tracking for off-road driving:
                Controller design, experimental validation and racing》
    :param vehicle:
    :param t_x: reference pose x
    :param t_y: reference pose y
    :param t_theta: reference pose yaw
    :param k: velocity gain
    :return: steer radius
    """
    phi = t_theta - vehicle.theta

    dx = t_x - vehicle.xf
    dy = t_y - vehicle.yf
    dist = np.sqrt(dx ** 2 + dy ** 2)
    e = np.sign(np.sin(np.arctan2(dy, dx) - vehicle.theta)) * dist

    delta = phi + np.arctan(k * e / (vehicle.v + 1e-5))

    delta_min = - np.pi / 6
    delta_max = np.pi / 6
    if delta > delta_max:
        delta = delta_max
    elif delta < delta_min:
        delta = delta_min

    return delta


def nearest_pose_index(vehicle, t_x_list, t_y_list):
    dx = [vehicle.xf - t_x for t_x in t_x_list]
    dy = [vehicle.yf - t_y for t_y in t_y_list]
    d = [np.abs(np.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    index = d.index(min(d))
    return index


def p_control(target, current, kp):
    a = kp * (target - current)
    return a


def main():
    k = 0.5
    Kp = 1
    T = 200
    target_velocity = 10 / 3.6

    t_x_list = [i * 0.1 for i in range(500)]
    # 直线
    # t_y_list = [i * 0 for i in range(0, 500, 1)]
    # t_theta_list = [i * 0 for i in range(0, 500, 1)]
    # 曲线
    t_y_list = [np.sin(t_x / 5.0) * t_x / 2.0 for t_x in t_x_list]
    t_theta_list = [np.arctan2(t_y_list[i + 1] - t_y_list[i], t_x_list[i + 1] - t_x_list[i])
                    for i in range(0, len(t_y_list) - 1, 1)]
    t_theta_list.append(t_theta_list[-1])

    vehicle = VehicleCF(0, -3, 0)

    last_index = len(t_x_list) - 1

    x = [vehicle.xf]
    y = [vehicle.yf]

    time = 0
    while time <= T:
        a = p_control(target_velocity, vehicle.v, Kp)
        target_index = nearest_pose_index(vehicle, t_x_list, t_y_list)

        delta = stanley_control(vehicle, t_x_list[target_index],
                                t_y_list[target_index],
                                t_theta_list[target_index], k)

        vehicle.step(vehicle.v, delta, a)

        time += vehicle.sample_time
        x.append(vehicle.xf)
        y.append(vehicle.yf)

        plt.cla()
        plt.plot(t_x_list, t_y_list, ".r", label="reference trajectory")
        plt.plot(x, y, "-b", label="vehicle trajectory")
        plt.plot(t_x_list[target_index], t_y_list[target_index], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Stanley Controller -- Speed [km/h]: " + str(vehicle.v * 3.6)[:4])
        plt.pause(0.001)

        if target_index >= last_index:
            break


if __name__ == "__main__":
    main()
