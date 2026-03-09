import numpy as np


def single_point_rect(lower: np.ndarray, upper: np.ndarray):
    """Returns a single point inside the region bounded by the lower left and upper right corner of the rectangle"""
    x_scale = upper[0] - lower[0]
    y_scale = upper[1] - lower[1]
    point = np.array([lower[0] + np.random.random() * x_scale, lower[1] + np.random.random() * y_scale])
    return point


def create_circle_2D(m : np.ndarray, r : int, n : int):
    """Returns a list of n points inside a circle with radius m around midpoint m"""
    points = np.zeros((n, 2))
    lower = np.array([m[0] - r, m[1] - r])
    upper = np.array([m[0] + r, m[1] + r])

    for i in range(n):
        point = single_point_rect(lower, upper)
        while np.linalg.norm(point - m) > r:
            point = single_point_rect(lower, upper)
        points[i] = point

    return points

"""
if __name__ != __main__:
    print("2D shape creator imported successfully")
    """