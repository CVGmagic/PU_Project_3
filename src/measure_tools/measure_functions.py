import numpy as np


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