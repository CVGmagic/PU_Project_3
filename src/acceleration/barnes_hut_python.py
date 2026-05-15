import numpy as np
from numba import njit, float64, int32, prange
from numba.experimental import jitclass
from numba.typed import List
from simulation.constants import G, eps_sq, epsilon
import math

# =========================
# GLOBAL PARAMETERS
# =========================

eps = epsilon
theta = 0.4

# =========================
# NODE STRUCTURE (replacement for C++ struct Node)
# =========================

node_spec = [
    ('mass', float64),
    ('com', float64[:]),  # center of mass (3,)
    ('center', float64[:]),  # center of cube (3,)
    ('half_size', float64),
    ('children', int32[:]),  # size 8
    ('particle', int32),
]


@jitclass(node_spec)
class Node:
    def __init__(self, center, half_size):
        self.mass = 0.0
        self.com = np.zeros(3, dtype=np.float64)
        self.center = center
        self.half_size = half_size
        self.children = np.full(8, -1, dtype=np.int32)
        self.particle = -1


# =========================
# BASIC VECTOR HELPERS
# =========================

@njit(fastmath=True)
def norm_sq(v):
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


# =========================
# OCTANT SELECTION
# =========================

@njit(fastmath=True)
def get_octant(node_center, particles, p_idx):
    rx = particles[p_idx, 0] - node_center[0]
    ry = particles[p_idx, 1] - node_center[1]
    rz = particles[p_idx, 2] - node_center[2]

    if rz >= 0:
        if rx >= 0 and ry >= 0: return 0
        if rx < 0 and ry >= 0: return 1
        if rx < 0 and ry < 0:  return 2
        return 3

    if rx >= 0 and ry >= 0: return 4
    if rx < 0 and ry >= 0: return 5
    if rx < 0 and ry < 0:  return 6
    return 7


# =========================
# CHILD CREATION
# =========================
@njit
def create_children(nodes, node_idx):
    node = nodes[node_idx]
    base = node.center
    hs = node.half_size / 2.0

    offsets = np.array([
        [1., 1., 1.],
        [-1., 1., 1.],
        [-1., -1., 1.],
        [1., -1., 1.],
        [1., 1., -1.],
        [-1., 1., -1.],
        [-1., -1., -1.],
        [1., -1., -1.],
    ], dtype=np.float64)

    for i in range(8):
        c = np.empty(3, dtype=np.float64)
        c[0] = base[0] + offsets[i, 0] * hs
        c[1] = base[1] + offsets[i, 1] * hs
        c[2] = base[2] + offsets[i, 2] * hs

        child = Node(c, hs)
        nodes.append(child)
        node.children[i] = len(nodes) - 1


# =========================
# INSERT INTO TREE
# =========================
@njit
def insert(nodes, particles, m, node_idx, p_idx):
    node = nodes[node_idx]

    if node.particle == -1 and node.children[0] == -1:
        node.particle = p_idx
        return

    if node.children[0] != -1:
        octant = get_octant(node.center, particles, p_idx)
        insert(nodes, particles, m, node.children[octant], p_idx)
        return

    old = node.particle
    node.particle = -1

    create_children(nodes, node_idx)

    insert(nodes, particles, m, node_idx, old)
    insert(nodes, particles, m, node_idx, p_idx)


# =========================
# MASS + CENTER OF MASS
# =========================
@njit
def set_mass_and_com(nodes, particles, m, node_idx):
    node = nodes[node_idx]

    if node.particle != -1:
        pidx = node.particle
        node.mass = m[pidx]
        node.com[0] = particles[pidx, 0]
        node.com[1] = particles[pidx, 1]
        node.com[2] = particles[pidx, 2]
        return

    node.mass = 0.0
    node.com[0] = 0.0
    node.com[1] = 0.0
    node.com[2] = 0.0

    if node.children[0] == -1:
        return

    for i in range(8):
        cidx = node.children[i]
        if cidx == -1:
            continue

        set_mass_and_com(nodes, particles, m, cidx)

        child = nodes[cidx]
        node.mass += child.mass

        # FIXED: Explicitly unrolled to avoid array allocation tracking
        node.com[0] += child.com[0] * child.mass
        node.com[1] += child.com[1] * child.mass
        node.com[2] += child.com[2] * child.mass

    if node.mass != 0.0:
        # FIXED: Scalar element division instead of node.com /= node.mass
        node.com[0] /= node.mass
        node.com[1] /= node.mass
        node.com[2] /= node.mass


# =========================
# ACCELERATION COMPUTATION
# =========================

@njit
def acceleration(nodes, particles, m, node_idx, p_idx, E_rel, theta):
    global eps_sq, G

    node = nodes[node_idx]

    if node.mass == 0.0 or node.particle == p_idx:
        return np.zeros(3, dtype=np.float64)

    dx = float(node.com[0] - particles[p_idx, 0])
    dy = float(node.com[1] - particles[p_idx, 1])
    dz = float(node.com[2] - particles[p_idx, 2])

    dist_sq = dx * dx + dy * dy + dz * dz
    dist = math.sqrt(dist_sq + eps_sq)

    inv_dist3 = 1.0 / ((dist_sq + eps_sq) * dist)

    if node.half_size / dist < theta and not is_in_node(node, p_idx, particles):
        f_grav = float(G * node.mass * inv_dist3)
        f_press = float(-E_rel * 1.0 / (dist_sq ** 4) / dist / m[p_idx])
        f_tot = f_grav + f_press

        force_tot = np.empty(3, dtype=np.float64)
        force_tot[0] = dx * f_tot
        force_tot[1] = dy * f_tot
        force_tot[2] = dz * f_tot
        return force_tot

    res = np.zeros(3, dtype=np.float64)

    for i in range(8):
        cidx = node.children[i]
        if cidx != -1:
            child_acc = acceleration(nodes, particles, m, cidx, p_idx, E_rel, theta)
            res[0] += child_acc[0]
            res[1] += child_acc[1]
            res[2] += child_acc[2]

    return res


# =========================
# MAIN FUNCTION
# =========================
@njit(parallel=True)
def compute_accelerations(particles, m, E_rel, theta=0.4):
    lower = np.zeros(3, dtype=np.float64)
    lower[0] = np.min(particles[:, 0])
    lower[1] = np.min(particles[:, 1])
    lower[2] = np.min(particles[:, 2])

    upper = np.zeros(3, dtype=np.float64)
    upper[0] = np.max(particles[:, 0])
    upper[1] = np.max(particles[:, 1])
    upper[2] = np.max(particles[:, 2])

    center = (lower + upper) / 2.0
    size = np.max(upper - lower) / 2.0 * 1.01 # To account for particles on the border

    nodes = List()
    root = Node(center, size)
    nodes.append(root)

    for i in range(len(particles)):
        insert(nodes, particles, m, 0, i)

    set_mass_and_com(nodes, particles, m, 0)

    acc = np.zeros_like(particles)

    for i in prange(len(particles)):
        acc[i] = acceleration(nodes, particles, m, 0, i, E_rel, theta)

    return acc


@njit
def is_in_node(node: Node, p_idx: int, particles: np.ndarray):
    return (
            node.center[0] - node.half_size <= particles[p_idx, 0] and
            node.center[0] + node.half_size > particles[p_idx, 0] and
            node.center[1] - node.half_size <= particles[p_idx, 1] and
            node.center[1] + node.half_size > particles[p_idx, 1] and
            node.center[2] - node.half_size <= particles[p_idx, 2] and
            node.center[2] + node.half_size > particles[p_idx, 2]
    )