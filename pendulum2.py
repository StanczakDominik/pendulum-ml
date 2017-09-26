import numpy as np
import matplotlib.pyplot as plt
import h5py
from pendulum import t, dt, m, l, g
from timeit import default_timer as timer
# import pycuda.gpuarray as gpuarray

timescale = np.sqrt(l / g)
dt = 0.1 / timescale
t = np.arange(0, 10000 * timescale, dt)

def prepare_initial_conditions(N_POINTS):
    theta1 = np.linspace(-3, 3, N_POINTS, endpoint=False)
    theta1, theta2 = np.meshgrid(theta1, theta1)
    p1 = np.zeros_like(theta1, dtype=np.float32)
    p2 = np.zeros_like(theta1, dtype=np.float32)
    r = np.dstack((theta1, theta2, p1, p2))
    flipped_point_iters = np.zeros_like(theta1, dtype=np.int32)
    return theta1, theta2, r, flipped_point_iters


def derivative(r):
    theta1, theta2, p1, p2 = np.dsplit(r, 4)
    common_factor = 6 / (m * l ** 2 * (16 - 9 * np.cos(theta1 - theta2)**2))
    dtheta1 = common_factor * (2 * p1 - 3 * np.cos(theta1 - theta2) * p2)
    dtheta2 = common_factor * (8 * p2 - 3 * np.cos(theta1 - theta2) * p1)
    common_factor2 = -0.5 * m * l**2
    dp1 = common_factor2 * (dtheta1 * dtheta2 * np.sin(theta1 - theta2) + 3 * g / l * np.sin(theta1))
    dp2 = common_factor2 * (-dtheta1 * dtheta2 * np.sin(theta1 - theta2) + g / l * np.sin(theta2))
    return np.dstack((dtheta1, dtheta2, dp1, dp2))


def iteration(r):
    k1 = derivative(r)
    k2 = derivative(r + k1 * dt * 0.5)
    k3 = derivative(r + k2 * dt * 0.5)
    k4 = derivative(r + k3 * dt)
    r += dt / 6 * (k1 + k4 + 2 * (k2 + k3))


def check_flipped(r):
    theta1 = r[:, :, 0]
    return (theta1 + np.pi) // (2 * np.pi) != 0


def still_going(flipped_point_iters):
    return flipped_point_iters == 0


def compute_loop(r, flipped_point_iters):
    for i, T in enumerate(t):
        iteration(r)
        have_flipped = check_flipped(r)
        are_still_going = still_going(flipped_point_iters)
        flipped_point_iters[have_flipped * are_still_going] = i

def main(N_POINTS = 200):
    theta1, theta2, r, flipped_point_iters = prepare_initial_conditions(N_POINTS)
    with h5py.File(f"pendulum_data_{N_POINTS}.hdf5") as f:
        f.create_dataset("theta1", data=theta1)
        f.create_dataset("theta2", data=theta2)
        f.create_dataset("t", data=t)

    
    start = timer()
    compute_loop(r, flipped_point_iters)
    timedelta = timer() - start
    print(f"Operation took {timedelta} s")

    flipped_point_iters = flipped_point_iters.astype(float)
    flipped_point_iters[still_going(flipped_point_iters)] = np.nan
    with h5py.File(f"pendulum_data_{N_POINTS}.hdf5") as f:
        f.create_dataset("flipped_point_iters", data=flipped_point_iters)

    plt.imshow(flipped_point_iters * dt)
    # plt.contourf(theta1, theta2, flipped_point_iters * dt, 50)
    plt.colorbar()
    plt.show()
        
        

if __name__ == "__main__":
    main(10)
