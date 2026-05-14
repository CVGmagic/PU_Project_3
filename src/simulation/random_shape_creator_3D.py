import numpy as np
from simulation.constants import epsilon

def single_point_cuboid(lower: np.ndarray, upper: np.ndarray):
    """Returns a single point inside the region bounded by the lower left and upper right corner of the cuboid"""
    if not np.all(upper >= lower):
        raise ValueError("All coordinates of lower must be smaller than or equal the corresponding coordinate of upper")

    x_scale = upper[0] - lower[0]
    y_scale = upper[1] - lower[1]
    z_scale = upper[2] - lower[2]
    point = np.array([lower[0] + np.random.random() * x_scale, lower[1] + np.random.random() * y_scale, lower[2] + np.random.random() * z_scale])
    return point


def create_cuboid_3D(lower: np.ndarray, upper: np.ndarray, n: int):
    points = np.zeros((n, 3))
    for i in range(n):
        points[i] = single_point_cuboid(lower, upper)
    return points


def create_sphere_3D(m : np.ndarray, r : int, n : int):
    """Returns a list of n points inside a circle with radius r around midpoint m"""
    points = np.zeros((n, 3))
    lower = m - r
    upper = m + r

    for i in range(n):
        point = single_point_cuboid(lower, upper)
        while np.linalg.norm(point - m) > r:
            point = single_point_cuboid(lower, upper)
        points[i] = point

    return points


def create_relaxed_sphere_3D(m : np.ndarray, positions : float, n : int) -> np.ndarray:
    """
    Creates a relaxed Sphere
    :param m: Midpoint of the Sphere
    :param positions: Radius of the initial Sphere
    :param n: Number of particles in the Sphere
    :returns: Positions of the particles
    """

    positions = create_sphere_3D(m, positions, n)
    mass = 10
    dt = 0.01
    v = np.full((n, 3), 0, dtype=float)
    eps_sq = epsilon * epsilon

    # Update acceleration
    diff = positions[:, None, :] - positions[None, :, :] # stores 3D-vector between every two-point combination
    dist_sq = np.sum(diff * diff, axis=-1) # stores 1D distance between evry two-point combination
    np.fill_diagonal(dist_sq, np.inf) # changes distance of two-point combination of same points to inf
    
    inv_dist_cubed = 1 / ((dist_sq + eps_sq) * np.sqrt(dist_sq + eps_sq))
    a = np.sum(diff * inv_dist_cubed[:, :, None], axis=1) / mass

    # Half velocity step
    v += a * dt / 2

    for i in range(3): # We do some number of timesteps
        positions += v * dt

        # Recompute acceleration
        diff = positions[:, None, :] - positions[None, :, :]  # stores 3D-vector between every two-point combination
        dist_sq = np.sum(diff * diff, axis=-1)  # stores 1D distance between evry two-point combination
        np.fill_diagonal(dist_sq, np.inf)  # changes distance of two-point combination of same points to inf

        inv_dist_cubed = 1 / ((dist_sq + eps_sq) * np.sqrt(dist_sq + eps_sq))
        a = np.sum(diff * inv_dist_cubed[:, :, None], axis=1) / mass

        # Update velocity
        v += a * dt

    # Find the furthest particle and scale the sphere back to the intended radius
    diff_m = positions - m
    dist_m_sq = np.sum(diff_m * diff_m, axis=1)
    mx_sq = np.max(dist_m_sq)
    mx = math.sqrt(mx_sq)
    # Maybe switch to some number of standard deviations instead
    positions *= (r / mx)

    return positions