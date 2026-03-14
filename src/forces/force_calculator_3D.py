import numpy as np
import math

# calculates how strong two points attract
def calculate_force(points: np.ndarray, a: int, b: int):
    r = math.sqrt(points[a]**2 + points[b]**2)
    single_force = G * m1 * m2 / r**3 * (points[b]-points[a])
    return single_force
# returns the force vector (uncomplete force)

# calculates the total force on a point 
def calculate_complete_force(points: np.ndarray, n: int):
    total_force = np.zeros(3)
    for i in points:
            if i != n:
                total_force += calculate_force(points, n, i)
    return total_force
# returns the total force on a single point as a vector

# creates an array with the total force of every point
def forces(points: np.ndarray):
    forces = np.zeros((len(points), 3))
    for i in forces:
        forces[i] = calculate_complete_force(points, i)
    return forces