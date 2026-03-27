import numpy as np
import math
from simulation.constants import G

# calculates how strong two points attract
def calculate_acc(points: np.ndarray, mass: np.ndarray, a: int, b: int):
    r = math.sqrt((points[a][0] - points[b][0])**2 + (points[a][1] - points[b][1])**2 + (points[a][2] - points[b][2])**2)
    single_acc = 1/(6.6743 * 10**(-11)) * mass[b]/ r**3 * (points[b]-points[a])
    # F = G * m1 * m2 / r**3 * (points[b]-points[a]), a = F / m1 = G * m2 / r**3 * (points[b]-points[a])
    return single_acc
# returns the acceleration vector (incomplete acceleration)


# calculates the total acceleration on a point
def calculate_complete_acc(points: np.ndarray, mass: np.ndarray, n: int):
    total_acc = np.zeros(3)

    for i in range(len(points)):
        if i != n:
            total_acc += calculate_acc(points, mass, n, i)

    return total_acc
# returns the total acceleration on a single point as a vector


# creates an array with the total acceleration of every point
def create_array_acc(points: np.ndarray, mass: np.ndarray):
    acc = np.zeros((len(points), 3))
    for i in range(len(acc)):
        acc[i] = calculate_complete_acc(points, mass, i)
    return acc


def calc_acc_rep_np(r, m):
    """Calculates a repulsive acceleration between point. The parameter m only serves to slow the acceleration"""
    eps = 1e-12 # Adds small "minimum" distance to prevent very large acceleration
    diff = r[:, None, :] - r[None, :, :] # stores 3D-vector between every two-point combination
    dist_sq = np.sum(diff * diff, axis=-1) + eps # stores 1D distance between evry two-point combination squared
    np.fill_diagonal(dist_sq, np.inf) # changes distance of two-point combination of same points to inf

    inv_dist_cubed = 1 / (dist_sq * np.sqrt(dist_sq)) # = 1/d^3
    a = np.sum(diff * inv_dist_cubed[:, :, None] * m[None, :, None], axis=1)
    return a


def calculate_attractive_acceleration(r, m):
    """
    Calculates an attractive acceleration between points.
    Has epsilon, but no short distance repulsion.
    """
    eps_sq = 1e-12 # Adds small "minimum" distance to prevent very large acceleration
    diff = r[:, None, :] - r[None, :, :]  # stores 3D-vector between every two-point combination
    dist_sq = np.sum(diff * diff, axis=-1) + eps_sq  # stores 1D distance between evry two-point combination squared
    np.fill_diagonal(dist_sq, np.inf)  # changes distance of two-point combination of same points to inf

    inv_dist_cubed = 1 / (dist_sq * np.sqrt(dist_sq))
    a = -np.sum(diff * inv_dist_cubed[:, :, None] * m[None, :, None], axis = 1)
    return a