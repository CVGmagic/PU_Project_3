import numpy as np
from simulation.constants import G, epsilon


def get_gravitational_energy(r, m) -> float:
    diff = r[:, None, :] - r[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) # Gets the lengths of all differences

    pairwise_mass = m[:, None] * m[None, :]
    pairwise_energy = -G * pairwise_mass / dist
    total_energy = np.sum(np.triu(pairwise_energy, k=1)) # Takes the sum of all elements above the diagonal (every pair is only included once and both particles have to be different)
    return total_energy


def get_kinetik_energy(m, v) -> float:
    return np.sum(np.sum(v * v, axis=-1) * m / 2)


def get_total_energy(r, m, v) -> float:
    return get_kinetik_energy(m, v) + get_gravitational_energy(r, m)