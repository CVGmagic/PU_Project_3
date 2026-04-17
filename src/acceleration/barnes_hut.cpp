#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
using namespace std;
namespace py = pybind11;


// When rebuilding, use the following commands in x64 native ...
// cd C:\Users\cvgma\VSCodeProjects\GitHub Projects\PU_Project_3\src\acceleration
// py setup.py build_ext --inplace


struct Vec3 { // My own vector class, to make calculations cleaner
    double x, y, z;

    // Define the most important operations for the vector
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s) const { return {x/s, y/s, z/s}; }
    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3& operator*=(double s) { x*=s; y*=s; z*=s; return *this; }
    Vec3& operator/=(double s) { x/=s; y/=s; z/=s; return *this; }
};


struct Node { // Node class to cleanly group data
    double mass;
    Vec3 com; // center of mass
    Vec3 center; // center of the cube
    double half_size;

    int children[8]; // Array of int with size 8 (indices of the 8 children)
    int particle; // Index of the particle, -1 if it's not a leaf
};


vector<Node> nodes; // Stores all the nodes. Root is at 0
vector<Vec3> particles; // Stores all the particles
vector<double> m; // Stores masses
double eps = 0.05; // Prevents infinite forces
double eps_sq = eps * eps;
double theta = 0.4; // Opening angle, controlls aggressiveness of pruning
double G = 6.6743e-11; // Gravitational constant


vector<Vec3> numpy_to_vec_vec3(py::array_t<double> positions) {
    // Converts the numpy array to vec3 for cleaner code
    auto positions_contig = py::array_t<double, py::array::c_style | py::array::forcecast>(positions);
    auto buf = positions_contig.request(); // Stores a buffer object, which basically contains all information about the array
    
    if (buf.ndim != 2 or buf.shape[1] != 3) {
        throw runtime_error("positions must have shape (N,3)");
    }

    int N = buf.shape[0];
    // ptr is a pointer to the first element in the array. This allows us to do pointer arithmetic
    double* ptr = static_cast<double*>(buf.ptr);

    vector<Vec3> result(N);
    for (int i = 0; i < N; i++) {
        Vec3 res;
        res.x = ptr[i*3 + 0];
        res.y = ptr[i*3 + 1];
        res.z = ptr[i*3 + 2];
        result[i] = res;
    }

    return result;
}


vector<double> numpy_to_vec(py::array_t<double> masses) {
    // Converts the numpy array to vec3 for cleaner code
    auto masses_contig = py::array_t<double, py::array::c_style | py::array::forcecast>(masses);
    auto buf = masses_contig.request(); // Stores a buffer object, which basically contains all information about the array
    
    if (buf.ndim != 1) {
        throw runtime_error("masses must have shape (N)");
    }

    int N = buf.shape[0];
    // ptr is a pointer to the first element in the array. This allows us to do pointer arithmetic
    double* ptr = static_cast<double*>(buf.ptr); 

    vector<double> result(N);
    for (int i = 0; i < N; i++) {
        result[i] = ptr[i];
    }

    return result;
}


py::array_t<double> vec3_to_numpy(vector<Vec3>& accelerations) {
    if (accelerations.empty()) {
        throw std::runtime_error("Empty acceleration vector");
    }
    
    size_t N = accelerations.size();

    // allocate new array
    py::array_t<double> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(N), 3});

    // Fill array with the computed values
    auto buf = result.request(); // Overwrite buf to change result instead

    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < N; i++) {
        ptr[i*3+0] = accelerations[i].x;
        ptr[i*3+1] = accelerations[i].y;
        ptr[i*3+2] = accelerations[i].z;
    }

    return result;
}

// function checks if has children return True, else return False
bool has_children(const Node& node) {
    return node.children[0] != -1;
}

// function checks whether a particle is in a specific node (also bigger ones)
bool is_in_node(const Node& node, int p_idx) {
    return (
        node.center.x - node.half_size <= particles[p_idx].x and
        node.center.x + node.half_size > particles[p_idx].x and
        node.center.y - node.half_size <= particles[p_idx].y and
        node.center.y + node.half_size > particles[p_idx].y and
        node.center.z - node.half_size <= particles[p_idx].z and
        node.center.z + node.half_size > particles[p_idx].z
    );
}

// connects a particle with its correct octant
int get_octant(Node& node, int p_idx) {
    Vec3& particle = particles[p_idx];

    Vec3 rel_cords = particle - node.center; // Moves coordinate system to node center

    // Top octants
    if (rel_cords.x >= 0 and rel_cords.y >= 0 and rel_cords.z >= 0) { // (0, 0, 0) is in Octant 0
        return 0;
    }
    if (rel_cords.x < 0 and rel_cords.y >= 0 and rel_cords.z >= 0) {
        return 1;
    }
    if (rel_cords.x < 0 and rel_cords.y < 0 and rel_cords.z >= 0) {
        return 2;
    }
    if (rel_cords.x > 0 and rel_cords.y < 0 and rel_cords.z >= 0) {
        return 3;
    }
    // Bottom octants
    if (rel_cords.x >= 0 and rel_cords.y >= 0 and rel_cords.z < 0) {
        return 4;
    }
    if (rel_cords.x < 0 and rel_cords.y >= 0 and rel_cords.z < 0) {
        return 5;
    }
    if (rel_cords.x < 0 and rel_cords.y < 0 and rel_cords.z < 0) {
        return 6;
    }
    if (rel_cords.x >= 0 and rel_cords.y < 0 and rel_cords.z < 0) {
        return 7;
    }
}


// This generates the child_nodes of a parent_node
void create_children(int node_idx) {
    cerr << "create children called with node_idx " << node_idx << "\n";
    // Caius needs this to debug but it does nothing

    nodes.reserve(nodes.size() + 8);
    /*the vector becomes much longer so it gets relocated -> pointers won't cause any troubles
    because the relocation is done before the scaling */

    Node& node = nodes[node_idx];

    Node child_0, child_1, child_2, child_3, child_4, child_5, child_6, child_7;

    child_0.center = node.center; // has the same position as parent_node
    child_0.half_size = node.half_size / 2; // has the half_length_size of parent_node
    // is set to have no children_nodes or particles in it
    child_0.particle = -1;
    fill(begin(child_0.children), end(child_0.children), -1);
    // adapts position of node
    child_0.center.x += child_0.half_size;
    child_0.center.y += child_0.half_size;
    child_0.center.z += child_0.half_size;

    // does the very same but for the 8 other points
    child_1.center = node.center;
    child_1.half_size = node.half_size / 2;
    child_1.particle = -1;
    fill(begin(child_1.children), end(child_1.children), -1);
    child_1.center.x -= child_1.half_size;
    child_1.center.y += child_1.half_size;
    child_1.center.z += child_1.half_size;
    

    child_2.center = node.center;
    child_2.half_size = node.half_size / 2;
    child_2.particle = -1;
    fill(begin(child_2.children), end(child_2.children), -1);
    child_2.center.x -= child_2.half_size;
    child_2.center.y -= child_2.half_size;
    child_2.center.z += child_2.half_size;
    

    child_3.center = node.center;
    child_3.half_size = node.half_size / 2;
    child_3.particle = -1;
    fill(begin(child_3.children), end(child_3.children), -1);
    child_3.center.x += child_3.half_size;
    child_3.center.y -= child_3.half_size;
    child_3.center.z += child_3.half_size;
    

    child_4.center = node.center;
    child_4.half_size = node.half_size / 2;
    child_4.particle = -1;
    fill(begin(child_4.children), end(child_4.children), -1);
    child_4.center.x += child_4.half_size;
    child_4.center.y += child_4.half_size;
    child_4.center.z -= child_4.half_size;
    

    child_5.center = node.center;
    child_5.half_size = node.half_size / 2;
    child_5.particle = -1;
    fill(begin(child_5.children), end(child_5.children), -1);
    child_5.center.x -= child_5.half_size;
    child_5.center.y += child_5.half_size;
    child_5.center.z -= child_5.half_size;
    

    child_6.center = node.center;
    child_6.half_size = node.half_size / 2;
    child_6.particle = -1;
    fill(begin(child_6.children), end(child_6.children), -1);
    child_6.center.x -= child_6.half_size;
    child_6.center.y -= child_6.half_size;
    child_6.center.z -= child_6.half_size;
    

    child_7.center = node.center;
    child_7.half_size = node.half_size / 2;
    child_7.particle = -1;
    fill(begin(child_7.children), end(child_7.children), -1);
    child_7.center.x += child_7.half_size;
    child_7.center.y -= child_7.half_size;
    child_7.center.z -= child_7.half_size;
    

    nodes.push_back(child_0);
    node.children[0] = nodes.size() - 1; // links the parent's first slot to child one

    nodes.push_back(child_1);
    node.children[1] = nodes.size() - 1; // links the parent's second slot to child two

    nodes.push_back(child_2);
    node.children[2] = nodes.size() - 1; // ...

    nodes.push_back(child_3);
    node.children[3] = nodes.size() - 1;

    nodes.push_back(child_4);
    node.children[4] = nodes.size() - 1;
    
    nodes.push_back(child_5);
    node.children[5] = nodes.size() - 1;

    nodes.push_back(child_6);
    node.children[6] = nodes.size() - 1;

    nodes.push_back(child_7);
    node.children[7] = nodes.size() - 1;
}


// --- THE INSERTION ALGORITHM ---
// This places a particle (p_idx) into the tree structure.
void insert(int node_idx, int p_idx) {
    cerr << "insert called with node_idx " << node_idx << " p_idx " << p_idx << "\n";

    // Insert particle p_idx into node node_idx
    Node& node = nodes[node_idx]; // Only store reference, avoid copying

    cerr << "node.particle = " << node.particle  << "\n";

/*
    cerr << "children:\n";
    for (int i = 0; i < 8; i++) {
        cerr << node.children[i] << " ";
    }
    cerr << "\n\n";
*/
// What do the lines above do except printing?

    // CASE 1: EMPTY LEAF
    // If the node is an empty leaf put the particle into the node
    if (node.particle == -1 and not has_children(node)) {
        cerr << "empty leaf\n";

        node.particle = p_idx;
        return;
    }

    // CASE 2: INTERNAL NODE (Already split)
    // If this node already has children, find out which child the particle belongs in.
    if (has_children(node)) { 
        cerr << "internal node\n";

        int oct = get_octant(node, p_idx); // Finds the correct child
        
        if (oct == -1 or oct > 7) {
            throw runtime_error("Invalid octant encountered");
        }

        insert(node.children[oct], p_idx); // Insert node into correct child
        return;
    }

    // CASE 3: LEAF WITH PARTICLE (Conflict!)
    // If we get here, the node is a leaf but ALREADY has a particle.
    // We must split this node into 8 smaller ones to accommodate both.
    cerr << "leaf with particle\n";

    int old_p_idx = node.particle; // Temporarily save the occupant.
    node.particle = -1; // This node is now a parent (no longer a leaf).

    create_children(node_idx); // creates the children
    cerr << "children created\n";

    // Re-insert both particles
    insert(node_idx, old_p_idx); // Insert old particle
    cerr << "old particle inserted\n";

    insert(node_idx, p_idx); // Insert new particle
    cerr << "new particle inserted\n";
}

// --- CALCULATING CENTER OF MASS ---
// Once the tree is built, we calculate the physics properties of each cube.
void set_mass_and_com(int node_idx) {
    Node& node = nodes[node_idx];

    // If it's a leaf with a particle, the mass/COM is just the particle itself.
    if (node.particle != -1) { // leaf with particle
        node.mass = m[node.particle];
        node.com = particles[node.particle];
        return;
    }

    // If it's an empty leaf, skip the rest
    if (not has_children(node)) {
        return;
    }

    node.mass = 0;
    node.com.x = 0;
    node.com.y = 0;
    node.com.z = 0;

    // Otherwise, initialize it as empty and sum up the children's properties.
    for (int i = 0; i < 8; i++) {
        if (node.children[i] == -1) { // if no particle in node, skip it
            continue;
        }

        set_mass_and_com(node.children[i]); // Update children first

        Node& child = nodes[node.children[i]];
        node.mass += child.mass; // Add child mass to total mass
        node.com += child.com * child.mass; // Shift center of mass (wrong but get's fixed later)
    }

    if (node.mass != 0) { // Prevent division by 0 for empty nodes
        // useless
        node.com /= node.mass; // Fix the scaling caused by the mass (basically a weighted average)
    }
}

// Calculates the magnitude (length) of a 3D vector: sqrt(x^2 + y^2 + z^2)
double norm(const Vec3& vec) {
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}
// Calculates the squared magnitude. This is a performance optimization:
// sqrt() is computationally expensive, so we use the squared distance whenever possible.
double norm_sq(const Vec3& vec) {
    return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}


Vec3 acceleration(int node_idx, int p_idx) {
    // Recursively computes the acceleration applied to particle p by all 
    // the particles inside the node

    Node& node = nodes[node_idx];

    // CHECK: Skip if the node is empty (mass=0)
    // or if the node IS the particle itself (a star doesn't pull on itself)
    if (node.mass == 0 or node.particle == p_idx) {
        Vec3 f;
        f.x = 0;
        f.y = 0;
        f.z = 0;
        return f; // Return zero acceleration
    }
    // else calculate acceleration like this:
    // Calculate displacement vector 'd' from the particle to the node's Center of Mass (com)
    Vec3 d = node.com - particles[p_idx];
    double dist_sq = norm_sq(d);
    double dist = sqrt(dist_sq + eps_sq); // softening parameter
    double inv_dist_cubed = 1 / ((dist_sq + eps_sq) * dist); // Gravity formula prep: 1 / r^3.

    // --- THE BARNES-HUT CRITERION ---
    // 'node.half_size / dist' is the "opening angle".
    // If the node is far away enough (smaller than 'theta'), we treat the
    // ENTIRE node as a single point mass. This is the big shortcut!
    if (node.half_size / dist < theta and not is_in_node(node, p_idx)) {
        return d * (G * node.mass * inv_dist_cubed);
    }

    // Otherwise, add the force of every single node (=particle)
    Vec3 f;
    f.x = 0;
    f.y = 0;
    f.z = 0;

    for (int i = 0; i < 8; i++) {
        if (node.children[i] == -1) {
            continue;
        }

        f += acceleration(node.children[i], p_idx);
    }

    return f;
}


// Gets the numpy array from python and returns the result. This function is what actually gets called by Python
py::array_t<double> compute_accelerations(py::array_t<double> positions, py::array_t<double> masses) {

    // 1. DATA CONVERSION: Move data from NumPy arrays to C++ vectors
    particles.clear();
    particles = numpy_to_vec_vec3(positions); // Get particles as vector<Vec3>
    
    cerr << "numpy_to_vec_vec3 ran\n";

    m.clear();
    m = numpy_to_vec(masses); // Assign masses to a global variable

    cerr << "numpy_to_vec ran\n";

    // 2. BOUNDING BOX: Find the limits of our universe to build the root node (first node)
    // We start with the largest/smallest possible doubles
    double lower_x = DBL_MAX, lower_y = DBL_MAX, lower_z = DBL_MAX;
    double upper_x = -DBL_MAX, upper_y = -DBL_MAX, upper_z = -DBL_MAX;

// find the smallest cube in which all particles are
    for (Vec3& p : particles) {
        lower_x = min(lower_x, p.x);
        lower_y = min(lower_y, p.y);
        lower_z = min(lower_z, p.z);

        upper_x = max(upper_x, p.x);
        upper_y = max(upper_y, p.y);
        upper_z = max(upper_z, p.z);
    }
   
    Vec3 root_center; // Center of the root node
    root_center.x = (lower_x + upper_x) / 2;
    root_center.y = (lower_y + upper_y) / 2;
    root_center.z = (lower_z + upper_z) / 2;

    double size_x = upper_x - lower_x;
    double size_y = upper_y - lower_y;
    double size_z = upper_z - lower_z;

    // Initialize root node
    Node root;
    root.center = root_center;
    root.half_size = max({size_x, size_y, size_z}) / 2;
    root.particle = -1;
    fill(begin(root.children), end(root.children), -1); // What does this do?

    nodes.clear(); // Clear from previous timestep
    nodes.push_back(root);

    cerr << "root initialized\n";

    // 4. TREE BUILDING: Insert every particle into the Octree (first biggest node and then get down)
    for (int i = 0; i < particles.size(); i++) {
        insert(0, i); // Insert particle into root node

        cerr << "insertion " << i << " ran successfully\n";
    }

    cerr << "tree built\n";

    // 5. AGGREGATION: Post-process the tree to calculate Mass and Center of Mass for all nodes
    set_mass_and_com(0);

    cerr << "masses set\n";

    // 6. CALCULATE PHYSICS: Compute acceleration for every single particle
    vector<Vec3> accelerations(particles.size());
    for (int i = 0; i < particles.size(); i++) {
        accelerations[i] = acceleration(0, i); // Calculates the acceleration of every particle to node 0 (big cube) and then scales down for detail

        cerr << "acceleration" << i <<" was set successfully\n";
    }
    // 7. RETURN: Convert C++ Vec3 results back to a NumPy array for Python
    py::array_t<double> res = vec3_to_numpy(accelerations);

    cerr << "result converted to numpy successfully\n";

    return res;
}


// Actually makes this a valid Python module
PYBIND11_MODULE(barnes_hut, module_object) {
    // Disable cerr
    std::cerr.setstate(std::ios_base::failbit);
    
    module_object.def("compute_accelerations", &compute_accelerations,
          "Compute accelerations using Barnes-Hut");
}
