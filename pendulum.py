import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import odeint
import dask
import dask.multiprocessing
import time

m = 1
l = 1
g = 1

r = np.array((np.pi/1, np.pi/3, 0, 0))
r = np.zeros(4)

timescale = np.sqrt(l / g)
t = np.arange(0, 1000 * timescale, 0.1/timescale)


def dtheta1(r):
    theta1, theta2, p1, p2 = r
    common_factor = 6 / (m * l ** 2 * (16 - 9 * np.cos(theta1 - theta2)**2))
    return common_factor * (2 * p1 - 3 * np.cos(theta1 - theta2) * p2)


def dtheta2(r):
    theta1, theta2, p1, p2 = r
    common_factor = 6 / (m * l ** 2 * (16 - 9 * np.cos(theta1 - theta2)**2))
    return common_factor * (8 * p2 - 3 * np.cos(theta1 - theta2) * p1)


def dp1(r):
    theta1, theta2, p1, p2 = r
    common_factor2 = -0.5 * m * l**2
    return common_factor2 * (dtheta1(r) * dtheta2(r) * np.sin(theta1 - theta2) + 3 * g / l * np.sin(theta1))


def dp2(r):
    theta1, theta2, p1, p2 = r
    common_factor2 = -0.5 * m * l**2
    return common_factor2 * (-dtheta1(r) * dtheta2(r) * np.sin(theta1 - theta2) + g / l * np.sin(theta2))


def derivative(r, t):
    return np.array((dtheta1(r), dtheta2(r), dp1(r), dp2(r)))


def energy(r):
    theta1, theta2, p1, p2 = r
    Dtheta1 = dtheta1(r)
    Dtheta2 = dtheta2(r)
    kinetic = m * l**2 / 6 * (Dtheta2**2 + 4 * Dtheta1**2 + 3 * Dtheta1 * Dtheta2 * np.cos(theta1 - theta2))
    potential = -0.5 * m * g * l * (3 * np.cos(theta1) + np.cos(theta2))
    return kinetic + potential


full_r = odeint(derivative, r, t)


def visualize(t, full_r):
    theta1, theta2, p1, p2 = full_r.T
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(t, theta1, label="theta1")
    ax1.plot(t, theta2, label="theta2")
    ax1.legend()
    ax2.plot(t, energy(full_r.T))
    ax2.set_title("Energy")
    plt.show()


def can_flip(theta1, theta2):
    """flips when this is positive"""
    return - 3 * np.cos(theta1) - np.cos(theta2) + 2


def visualize_energy_surplus(theta_range=np.linspace(-np.pi, np.pi, 300)):
    THETA1, THETA2 = np.meshgrid(theta_range, theta_range)
    flips_at = can_flip(THETA1, THETA2)
    plt.contourf(THETA1, THETA2, flips_at, 50)
    plt.colorbar()
    plt.contour(THETA1, THETA2, flips_at >= 0, 1)
    plt.title("Energy surplus")
    plt.show()


def run_simulation(r, t):
    full_r = odeint(derivative, r, t)
    return when_does_flip(t, full_r)


def create_data(n_points=30):
    with h5py.File("pendulum_data_small.hdf5") as f:
        if "t" not in f:
            f.create_dataset("t", data=t)
        theta_range = np.linspace(-np.pi, np.pi, n_points)
        theta = np.array([(theta1, theta2, 0, 0) for theta1 in theta_range for theta2 in theta_range])
        f.create_dataset("theta", data=theta)
        values = [dask.delayed(run_simulation)(r, t) for r in theta]
        results = dask.compute(*values, get=dask.multiprocessing.get)
        f.create_dataset("iters_at_flip", data=results)


def does_flip(t, full_r):
    theta1, theta2, p1, p2 = full_r.T
    theta1 = (theta1 + np.pi) // (2 * np.pi)
    return np.any(theta1 != 0)
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.plot(t, theta1, label="theta1")
    # ax1.legend()
    # ax2.plot(t, energy(full_r.T))
    # ax2.set_title("Energy")
    # plt.show()


def when_does_flip(t, full_r):
    theta1, theta2, p1, p2 = full_r.T
    theta1 = (theta1 + np.pi) // (2 * np.pi)
    nonzero_flip = np.argmax(theta1 != 0)
    if nonzero_flip == 0:
        nonzero_flip = len(t)
    return nonzero_flip


def load_data():
    with h5py.File("pendulum_data_small.hdf5") as f:
        theta = f['theta'][...]
        theta1 = theta[:, 0]
        theta2 = theta[:, 1]
        flips = f['iters_at_flip'][...]

    theta1_range = np.linspace(theta1.min(), theta1.max(), 100)
    theta2_range = np.linspace(theta2.min(), theta2.max(), 100)
    THETA1, THETA2 = np.meshgrid(theta1_range, theta2_range)
    flips_at = can_flip(THETA1, THETA2)
    plt.contourf(THETA1, THETA2, flips_at, 50)
    plt.colorbar()
    plt.contour(THETA1, THETA2, flips_at >= 0, 1)
    plt.title("Energy surplus")
    plt.scatter(theta1, theta2, c=flips)
    plt.colorbar()
    plt.show()


def continuous_create():
    n_points = 10
    while True:
        print(f"Running for {n_points} points within the range")
        start_time = time.time()
        create_data(n_points)
        run_time = time.time() - start_time
        print(f"Total run time in this pass: {run_time}")
        n_points *= 2


if __name__ == "__main__":
    try:
        load_data()
    except KeyError:
        print("File doesn't exist yet!")
    continuous_create()
