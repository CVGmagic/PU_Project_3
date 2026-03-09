import numpy as np


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
    """Returns a list of n points inside a circle with radius m around midpoint m"""
    points = np.zeros((n, 3))
    lower = m - r
    upper = m + r

    for i in range(n):
        point = single_point_cuboid(lower, upper)
        while np.linalg.norm(point - m) > r:
            point = single_point_cuboid(lower, upper)
        points[i] = point

    return points


