import numpy as np
from typing import Any

"""
This file gives the type hints and autocomplete options for barnes_hut.
It does not affect the actual computation
"""

def compute_accelerations(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Compute accelerations on N particles using the Barnes-Hut algorithm.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3), giving the x, y, z coordinates of each particle.
    masses : np.ndarray
        Array of shape (N,), giving the mass of each particle.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3), containing the computed acceleration vectors
        for each particle.
    """
    ...