import numpy as np
import math

# calculates how strong two points attract
def calculate_acc(points: np.ndarray, mass: np.ndarray, a: int, b: int):
    r = math.sqrt(points[a]**2 + points[b]**2)
    single_acc = 1/(6.6743 * 10**(-11)) * mass[b]/ r**3 * (points[b]-points[a])
    # F = G * m1 * m2 / r**3 * (points[b]-points[a]), a = F / m1 = G * m2 / r**3 * (points[b]-points[a])
    return single_acc
# returns the acceleration vector (incomplete acceleration)

# calculates the total acceleration on a point
def calculate_complete_acc(points: np.ndarray, mass: np.ndarray, n: int):
    total_acc = np.zeros(3)
    for i in points:
        if i != n:
            total_acc += calculate_acc(points, mass, n, i)
    return total_acc
# returns the total acceleration on a single point as a vector

# creates an array with the total acceleration of every point
def create_array_acc(points: np.ndarray, mass: np.ndarray):
    acc = np.zeros((len(points), 3))
    for i in acc:
        acc[i] = calculate_complete_acc(points, mass, i)
    return acc