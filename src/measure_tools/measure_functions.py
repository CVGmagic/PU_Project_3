import numpy as np
import time
from acceleration.acceleration_calculator_3D import calculate_gravitational_acceleration
from acceleration.barnes_hut_python import compute_accelerations
import matplotlib.pyplot as plt
from simulation.random_shape_creator_3D import create_relaxed_sphere_3D
import math
from simulation.constants import G
from setup import generate_points


def compute_covariance_matrix(positions) -> np.ndarray:
    """Computes the covariance matrix with planet centered coordinates"""
    # These are now cloud centered coords
    X = positions[1:] - np.mean(positions[1:], axis=0) # Exclude particle 0 (the sun)

    cov_matrix = X.T @ X / len(X)

    return cov_matrix


def compute_principal_axis_lengths(positions) -> np.ndarray:
    """
    :param positions: An array containing the positions
    :returns: An ndarray of length 3 containing the principal axis lengths
    """
    cov_matrix = compute_covariance_matrix(positions)

    return np.linalg.eigvalsh(cov_matrix)


def compute_covariance_matrix_outliers_excluded(positions, threshold=2.5) -> np.ndarray:
    """
    Computes the covariance matrix with planet centered coordinates excluding outliers

    :param positions: Positions of all particles, sun is expected to be at idx 0
    :param threshold: Defines how many standard deviations away a particle can be before being excluded
    """
    # These are now cloud centered coords
    X = positions[1:] - np.mean(positions[1:], axis=0)  # Exclude particle 0 (the sun)

    dist = np.linalg.norm(X, axis=1)
    sigma = np.sqrt(np.sum((dist - np.mean(dist)) ** 2))

    mask = np.where(dist <= threshold * sigma)
    X_filtered = X[mask]

    cov_matrix = X_filtered.T @ X_filtered / len(X_filtered)

    return cov_matrix


def compute_principal_axis_lengths_outliers_excluded(positions, threshold=2.5):
    """
        :param positions: An array containing the positions
        :returns: An ndarray of length 3 containing the principal axis lengths
    """
    cov_matrix = compute_covariance_matrix_outliers_excluded(positions, threshold)

    return np.linalg.eigvalsh(cov_matrix)


def compute_elongation(positions, threshold=2.5) -> float:
    principal_axis_lengths = compute_principal_axis_lengths_outliers_excluded(r)
    np.sort(principal_axis_lengths)
    return principal_axis_lengths[2] / principal_axis_lengths[0]


def benchmark_computation(particle_counts: list[int], do_standard=True, do_barnes_hut=True, opening_angles=[0.4], trials=100, dt=0.0001):
    n = len(particle_counts)
    standard_times = np.empty(n)
    barnes_hut_times = np.empty((n, len(opening_angles)))
    r_init = create_relaxed_sphere_3D()
    v_init = np.random.random((n, 3)) * 15
    m_init = np.random.random(n) * 15

    for i, particle_count in enumerate(particle_counts):
        if do_standard:
            r = np.copy(r_init)
            v = np.copy(v_init)
            m = np.copy(m_init)
            # Warmup call
            calculate_gravitational_acceleration(r, m, 0.5)

            start_time = time.perf_counter()
            for _ in range(trials):
                a = calculate_gravitational_acceleration(r, m, 0.5)
                v += a * dt
                r += v * dt
            end_time = time.perf_counter()
            standard_times[i] = end_time - start_time

        if do_barnes_hut:
            angle_times = np.zeros(len(opening_angles))
            for j, theta in enumerate(opening_angles):
                r = np.copy(r_init)
                v = np.copy(v_init)
                m = np.copy(m_init)
                # Warmup call
                compute_accelerations(r, m, 0.5)

                start_time = time.perf_counter()
                for _ in range(trials):
                    a = compute_accelerations(r, m, 0.5)
                    v += a * dt
                    r += v * dt
                end_time = time.perf_counter()
                angle_times[j] = end_time - start_time
            barnes_hut_times[i] = angle_times

    plt.figure()
    plt.title(f"Comparison between Standard and Barnes Hut Algorithm on {trials} timesteps with dt = {dt}")
    plt.xlabel("Number of particles")
    plt.ylabel("Elapsed time [s]")
    plt.plot(particle_counts, standard_times, label="Standard Times")
    for i, theta in enumerate(opening_angles):
        plt.plot(particle_counts, barnes_hut_times[:, i], label=f"Barnes hut, theta = {theta}")
    plt.legend(loc="best")
    plt.show()


def planet_com(positions, masses) -> np.ndarray:
    return np.sum(positions[1:] * masses[1:].reshape(-1, 1), axis=0) / np.sum(masses[1:])


def circular_orbit_velocity(dist_star, m_star) -> float:
    global G
    return math.sqrt(G * m_star / dist_star)


def roche_limit(m_planet, m_star, r_planet) -> float:
    return 2.44 * r_planet * (m_star / m_planet)**(1/3)


def plot_elongations(distances: np.ndarray, n=500, dt=0.001, m_planet=50_000, m_star: float=None, timesteps=1000):
    """
    Takes distances as a fraction of the roche limit and plots the axis elongation over some number of timesteps
    :returns: Function unfinished, returns None for now
    """
    n = 500
    dt = 0.0001
    mass = m_planet / n
    if m_star is None:
        m_star = mass_planet * 333_000  # Sun earth ratio
    roche = roche_limit(m_planet, m_star, r_planet)

    elongations = np.empty((len(distances), timesteps))
    for distance in distances:
        distance_star = distance * roche
        v_rotation = circular_orbit_velocity(distance_star, mass_star)

    # TODO Maybe finish this function
    return



