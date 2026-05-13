import numpy as np


def compute_covariance_matrix(positions):
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


