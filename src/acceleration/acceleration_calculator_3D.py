import numpy as np
from numba import njit, prange
import math
from simulation.constants import G, eps_sq, r0, k

"""
sigma = 0.5   # The 'size' of your particles.
epsilon = 0.1 # The 'strength' of the bounce.
k = 4 * epsilon * (sigma**12)

# calculates how strong two points attract
def calculate_acc(points: np.ndarray, mass: np.ndarray, a: int, b: int):
    r = math.sqrt((points[a][0] - points[b][0])**2 + (points[a][1] - points[b][1])**2 + (points[a][2] - points[b][2])**2)
    single_acc = ((6.6743 * 10**(-11)) * mass[b]/ r**3 - k/r**13)* (points[b]-points[a])
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

"""

def calc_acc_rep_np(r, m):
    """Calculates a repulsive acceleration between point. The parameter m only serves to slow the acceleration"""
    diff = r[:, None, :] - r[None, :, :] # stores 3D-vector between every two-point combination
    dist_sq = np.sum(diff * diff, axis=-1) # stores 1D distance between evry two-point combination squared
    np.fill_diagonal(dist_sq, np.inf) # changes distance of two-point combination of same points to inf

    inv_dist_cubed = 1 / ((dist_sq + eps_sq) * np.sqrt(dist_sq + eps_sq))
    a = np.sum(diff * inv_dist_cubed[:, :, None] * m[None, :, None], axis=1)
    return a


@njit(fastmath=True, parallel=True)
def calculate_gravitational_acceleration(r, m, en_rel):
    """
    Calculates an attractive acceleration between points.
    Has epsilon, but no short distance repulsion.
    """
    global eps_sq, G

    n = r.shape[0]

    acc = np.zeros_like(r)
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue

            diff = r[j] - r[i]

            dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

            """Calculate Gravity"""
            inv_dist_3 = 1.0 / math.sqrt(dist_sq + eps_sq)**3

            a_grav = diff * m[j] * G * inv_dist_3

            """Calculate Pressure"""
            a_pressure = -en_rel * diff / (dist_sq)**4.5 / m[i]

            acc[i] += a_pressure + a_grav

    return acc

    """
    diff = r[:, None, :] - r[None, :, :]  # stores 3D-vector between every two-point combination
    dist_sq = np.sum(diff * diff, axis=-1)  # stores 1D distance between evry two-point combination squared
    np.fill_diagonal(dist_sq, np.inf)  # changes distance of two-point combination of same points to inf

    inv_dist_3 = 1 / ((dist_sq + eps_sq) * np.sqrt(dist_sq + eps_sq))

    a_pressure = diff * -en_rel / dist_sq**4 / np.sqrt(dist_sq)
    a_grav = -np.sum(diff * inv_dist_3[:, :, None] * m[None, :, None], axis = 1)

    return a_grav + a_pressure
    """



def calculate_gravitational_acceleration_caius(r, m):
    """
    Calculates an attractive acceleration between points.
    Has epsilon, but no short distance repulsion.
    """
    global G, eps_sq, k, r0

    diff = r[:, None, :] - r[None, :, :]  # stores 3D-vector between every two-point combination
    dist_sq = np.sum(diff * diff, axis=-1)  # stores 1D distance between evry two-point combination squared
    np.fill_diagonal(dist_sq, np.inf)  # changes distance of two-point combination of same points to inf

    inv_dist_3 = 1 / ((dist_sq + eps_sq) * np.sqrt(dist_sq + eps_sq))

    a = -np.sum(diff * inv_dist_3[:, :, None] * m[None, :, None], axis = 1) * G

    return a

