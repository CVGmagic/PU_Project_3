import numpy as np
import math
from simulation.constants import eps_sq
from acceleration.acceleration_calculator_3D import calculate_gravitational_acceleration
from simulation.energy_calculator import calculate_energy_relation


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


def create_relaxed_sphere_3D(m : np.ndarray, r : float, n : int, expansion_steps: int = 3, contraction_steps: int = 10) -> np.ndarray:
    """
    Creates a relaxed Sphere
    :param m: Midpoint of the Sphere
    :param r: Radius of the initial Sphere
    :param n: Number of particles in the Sphere
    :param relaxation_steps: How many steps of repulsive acceleration to do
    :returns: Positions of the particles
    """

    positions = create_sphere_3D(m, r, n)
    dt = 0.0001
    mass = np.ones(n, dtype=np.float64)

    """Expansion"""
    a = -calculate_gravitational_acceleration(positions, mass, 0)
    v = a * (dt / 2)

    for i in range(expansion_steps): # We do some number of timesteps
        positions += v * dt

        a = -calculate_gravitational_acceleration(positions, mass, 0)
        v += a * dt

        v *= 0.9 # Damping

    """Contraction"""
    e_rel = calculate_energy_relation(positions, mass)
    a = calculate_gravitational_acceleration(positions, mass, e_rel)
    v = a * (dt / 2)

    for i in range(contraction_steps):
        positions += v * dt

        a = calculate_gravitational_acceleration(positions, mass, e_rel)
        v += a * dt

    # Find the furthest particle and scale the sphere back to the intended radius
    diff_m = positions - m
    dist_m_sq = np.sum(diff_m * diff_m, axis=1)
    mx_sq = np.max(dist_m_sq)
    mx = math.sqrt(mx_sq)
    # Maybe switch to some number of standard deviations instead
    positions *= (r / mx)

    return positions