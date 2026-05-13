import numpy as np
import time
from acceleration.acceleration_calculator_3D import calculate_gravitational_acceleration
from acceleration.barnes_hut_python import compute_accelerations
import matplotlib.pyplot as plt


def compute_covariance_matrix(positions) -> np.ndarray:
    """Computes the covariance matrix with planet centered coordinates"""
    # These are now cloud centered coords
    X = positions - np.mean(positions[1:], axis=0) # Exclude particle 0 (the sun)

    cov_matrix = X.T @ X / len(X)

    return cov_matrix


def compute_principal_axis_lengths(positions) -> np.ndarray:
    """
    :param positions: An array containing the positions
    :returns: An ndarray of length 3 containing the principal axis lengths
    """
    cov_matrix = compute_covariance_matrix(positions)

    return np.linalg.eigvalsh(cov_matrix)


def compute_covariance_matrix_outliers_excluded(positions, threshold=3) -> np.ndarray:
    """
    Computes the covariance matrix with planet centered coordinates excluding outliers

    :param positions: Positions of all particles, sun is expected to be at idx 0
    :param threshold: Defines how many standard deviations away a particle can be before being excluded
    """
    # These are now cloud centered coords
    X = positions - np.mean(positions[1:], axis=0)  # Exclude particle 0 (the sun)

    dist = np.linalg.norm(X, axis=1)
    sigma = np.sqrt(np.sum((dist - np.mean(dist)) ** 2))

    mask = np.where(dist <= threshold * sigma)
    X_filtered = X[mask]

    cov_matrix = X_filtered.T @ X_filtered / len(X_filtered)

    return cov_matrix


def compute_principal_axis_lengths_outliers_excluded(positions, threshold=3):
    """
        :param positions: An array containing the positions
        :returns: An ndarray of length 3 containing the principal axis lengths
    """
    cov_matrix = compute_covariance_matrix_outliers_excluded(positions, threshold)

    return np.linalg.eigvalsh(cov_matrix)


def benchmark_computation(particle_counts: list[int], do_standard=True, do_barnes_hut=True, opening_angles=[0.4], trials=100, dt=0.0001):
    n = len(particle_counts)
    standard_times = np.empty(n)
    barnes_hut_times = np.empty((n, len(opening_angles)))
    r_init = np.random.random((n, 3)) * 15
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

