import numpy as np
from numba import njit, float64, int32
from numba.experimental import jitclass
from numba.typed import List

# =========================
# GLOBAL PARAMETERS
# =========================

eps = 0.05
eps_sq = eps * eps
theta = 0.4
G = 6.6743e-11


# =========================
# NODE STRUCTURE (replacement for C++ struct Node)
# =========================

node_spec = [
    ('mass', float64),
    ('com', float64[:]),        # center of mass (3,)
    ('center', float64[:]),     # center of cube (3,)
    ('half_size', float64),
    ('children', int32[:]),     # size 8
    ('particle', int32),
]


@jitclass(node_spec)
class Node:
    def __init__(self, center, half_size):
        # Stores all the node data
        self.mass = 0.0
        self.com = np.zeros(3)
        self.center = center
        self.half_size = half_size
        self.children = np.full(8, -1, dtype=np.int32)
        self.particle = -1


# =========================
# BASIC VECTOR HELPERS
# =========================

@njit(fastmath=True)
def norm_sq(v):
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]


# =========================
# OCTANT SELECTION
# =========================

@njit(fastmath=True)
def get_octant(node_center, particle_pos):
    # Moves coordinate system to node center
    rx = particle_pos[0] - node_center[0]
    ry = particle_pos[1] - node_center[1]
    rz = particle_pos[2] - node_center[2]

    # Top 4 octants (z >= 0)
    if rz >= 0:
        if rx >= 0 and ry >= 0:
            return 0
        if rx < 0 and ry >= 0:
            return 1
        if rx < 0 and ry < 0:
            return 2
        return 3

    # Bottom 4 octants (z < 0)
    if rx >= 0 and ry >= 0:
        return 4
    if rx < 0 and ry >= 0:
        return 5
    if rx < 0 and ry < 0:
        return 6
    return 7


# =========================
# CHILD CREATION
# =========================

def create_children(nodes, node_idx):
    # IMPORTANT: Python-level list expansion (Numba limitation)
    node = nodes[node_idx]

    base = node.center
    hs = node.half_size / 2

    offsets = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
        [ 1, -1, -1],
    ], dtype=np.float64)

    for i in range(8):
        c = base + offsets[i] * hs
        child = Node(c, hs)
        nodes.append(child)
        node.children[i] = len(nodes) - 1


# =========================
# INSERT INTO TREE
# =========================

def insert(nodes, particles, m, node_idx, p_idx):

    node = nodes[node_idx]

    # Empty leaf
    if node.particle == -1 and node.children[0] == -1:
        node.particle = p_idx
        return

    # Internal node
    if node.children[0] != -1:
        octant = get_octant(node.center, particles[p_idx])
        insert(nodes, particles, m, node.children[octant], p_idx)
        return

    # Leaf with particle → subdivide
    old = node.particle
    node.particle = -1

    create_children(nodes, node_idx)

    insert(nodes, particles, m, node_idx, old)
    insert(nodes, particles, m, node_idx, p_idx)


# =========================
# MASS + CENTER OF MASS
# =========================

def set_mass_and_com(nodes, particles, m, node_idx):

    node = nodes[node_idx]

    if node.particle != -1:
        node.mass = m[node.particle]
        node.com[:] = particles[node.particle]
        return

    node.mass = 0.0
    node.com[:] = 0.0

    if node.children[0] == -1:
        return

    for i in range(8):
        cidx = node.children[i]
        if cidx == -1:
            continue

        set_mass_and_com(nodes, particles, m, cidx)

        child = nodes[cidx]
        node.mass += child.mass
        node.com += child.com * child.mass

    if node.mass != 0:
        node.com /= node.mass


# =========================
# ACCELERATION COMPUTATION
# =========================

@njit(fastmath=True)
def acceleration(nodes, particles, node_idx, p_idx):

    node = nodes[node_idx]

    # No mass or same particle
    if node.mass == 0.0 or node.particle == p_idx:
        return np.zeros(3)

    dx = node.com[0] - particles[p_idx, 0]
    dy = node.com[1] - particles[p_idx, 1]
    dz = node.com[2] - particles[p_idx, 2]

    dist_sq = dx*dx + dy*dy + dz*dz
    dist = np.sqrt(dist_sq + eps_sq)

    inv_dist3 = 1.0 / ((dist_sq + eps_sq) * dist)

    # Barnes–Hut approximation condition
    if node.half_size / dist < theta and node.children[0] == -1:
        f = G * node.mass * inv_dist3
        return np.array([dx*f, dy*f, dz*f])

    # Otherwise recurse
    res = np.zeros(3)

    for i in range(8):
        cidx = node.children[i]
        if cidx != -1:
            res += acceleration(nodes, particles, cidx, p_idx)

    return res


# =========================
# MAIN FUNCTION
# =========================

def compute_accelerations(positions, masses):

    particles = positions.astype(np.float64)
    m = masses.astype(np.float64)

    # Bounding box
    lower = np.min(particles, axis=0)
    upper = np.max(particles, axis=0)

    center = (lower + upper) / 2
    size = np.max(upper - lower) / 2

    # Root node
    nodes = List()
    root = Node(center, size)
    nodes.append(root)

    # Build tree
    for i in range(len(particles)):
        insert(nodes, particles, m, 0, i)

    # Mass + COM pass
    set_mass_and_com(nodes, particles, m, 0)

    # Compute accelerations
    acc = np.zeros_like(particles)

    for i in range(len(particles)):
        acc[i] = acceleration(nodes, particles, 0, i)

    return acc