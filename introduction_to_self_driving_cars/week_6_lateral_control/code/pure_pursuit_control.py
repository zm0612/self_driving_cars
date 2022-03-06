import numpy as np
import matplotlib.pyplot as plt


class VehicleRF:
    def __init__(self, xr=0, yr=0, theta=0, v=0):
        self.xr = xr
        self.yr = yr
        self.theta = theta
        self.v = v

        self.L = 2.9
        self.sample_time = 0.1

    def step(self, delta, a):
        self.xr += self.v * np.cos(self.theta) * self.sample_time
        self.yr += self.v * np.sin(self.theta) * self.sample_time
        self.theta += self.v * np.tan(delta) / self.L * self.sample_time
        self.v += a * self.sample_time


def look_ahead_point_index(vehicle, t_x_list, t_y_list, k, LFC):
    dx = [vehicle.xr - t_x for t_x in t_x_list]
    dy = [vehicle.yr - t_y for t_y in t_y_list]
    d = [np.abs(np.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    index = d.index(min(d))
    L = 0

    LF = k * vehicle.v + LFC

    while LF > L and (index + 1) < len(t_x_list):
        dx = t_x_list[index + 1] - t_x_list[index]
        dy = t_y_list[index + 1] - t_y_list[index]
        L += np.sqrt(dx ** 2 + dy ** 2)
        index += 1

    return index


def p_control(target, current, kp):
    a = kp * (target - current)
    return a


def pure_pursuit_control(vehicle, t_x, t_y, k, LFC=0.0):
    alpha = np.arctan2(t_y - vehicle.yr, t_x - vehicle.xr) - vehicle.theta
    delta = np.arctan(2 * vehicle.L * np.sin(alpha) / (k * (vehicle.v + 1e-5) + LFC))

    delta_min = -np.pi / 6
    delta_max = np.pi / 6
    if delta > delta_max:
        delta = delta_max
    elif delta < delta_min:
        delta = delta_min

    return delta


def main():
    LFC = 2.0
    k = 0.1
    Kp = 1
    T = 200
    target_velocity = 10 / 3.6

    t_x_list = [i * 0.1 for i in range(500)]
    # 直线
    # t_y_list = [i * 0 for i in range(500)]
    # 曲线
    t_y_list = [np.sin(t_x / 5.0) * t_x / 2.0 for t_x in t_x_list]

    vehicle = VehicleRF(xr=0, yr=-3, theta=0, v=0)

    x = [vehicle.xr]
    y = [vehicle.yr]
    time = 0

    last_index = len(t_x_list) - 1

    while time <= T:
        a = p_control(target_velocity, vehicle.v, Kp)
        target_index = look_ahead_point_index(vehicle, t_x_list, t_y_list, k, LFC)
        delta = pure_pursuit_control(vehicle, t_x_list[target_index],
                                     t_y_list[target_index], k, LFC)

        vehicle.step(delta, a)

        time += vehicle.sample_time
        x.append(vehicle.xr)
        y.append(vehicle.yr)

        plt.cla()
        plt.plot(t_x_list, t_y_list, ".r", label="reference trajectory")
        plt.plot(x, y, "-b", label="vehicle trajectory")
        plt.plot(t_x_list[target_index], t_y_list[target_index], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Pure Pursuit Controller -- Speed [km/h]: " + str(vehicle.v * 3.6)[:4])
        plt.pause(0.001)

        if target_index >= last_index:
            break


if __name__ == "__main__":
    main()
