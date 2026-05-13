import numpy as np
from simulation.constants import G, epsilon, eps_sq


def get_gravitational_energy(r, m) -> float:
    diff = r[:, None, :] - r[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) # Gets the lengths of all differences

    pairwise_mass = m[:, None] * m[None, :]
    pairwise_energy = -G * pairwise_mass / (dist + epsilon)
    total_energy = np.sum(np.triu(pairwise_energy, k=1)) # Takes the sum of all elements above the diagonal (every pair is only included once and both particles have to be different)
    return total_energy


def get_new_gravitational_energy(r, m) -> float:
    diff = r[:, None, :] - r[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # Gets the lengths of all differences

    pairwise_mass = m[:, None] * m[None, :]
    pairwise_energy = -G * pairwise_mass * 2 * (1 / epsilon**2 - 1 / dist**3)
    total_energy = np.sum(np.triu(pairwise_energy, k=1))
    return total_energy


def get_kinetik_energy(m, v) -> float:
    return np.sum(np.sum(v * v, axis=-1) * m / 2)


def get_total_energy(r, m, v) -> float:
    return get_kinetik_energy(m, v) + get_gravitational_energy(r, m)


def calculate_potential_energies(r, m):
    """
    Calculates Gravity PE and Pressure PE in one pass.
    :returns: (gravity_potential_energy, pressure_potential_energy)
    """
    # 1. Core distance calculations
    diff = r[:, None, :] - r[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1) + eps_sq
    np.fill_diagonal(dist_sq, np.inf)

    # 2. Shared variables
    dist = np.sqrt(dist_sq)
    mass_matrix = m[:, None] * m[None, :]

    # 3. Gravity Energy (Integral of 1/r^2 is -1/r)
    # Using the 0.5 here to account for double-counting pairs
    gravity_potential_energy = -0.5 * np.sum(mass_matrix * (1 / dist)) * G

    # 4. Pressure Energy (Integral of 1/r^8 is 1/(7 * r^7))
    # r^7 = (dist_sq^3 * dist)
    pressure_potential_energy = 0.5 * np.sum(mass_matrix * (1 / (7 * dist_sq ** 3 * dist)))

    return gravity_potential_energy, pressure_potential_energy

