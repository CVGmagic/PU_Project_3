#include <bits/stdc++.h>
#define int int64_t
using namespace std;


vector<Node> nodes; // Stores all the nodes. Root is at 0
vector<Vec3> particles; // Stores all the particles
vector<int> m; // Stores masses


struct Vec3 { // My own vector class, to make calculations cleaner
    int x, y, z;

    // Define the most important operations for the vector
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(int s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(int s) const { return {x/s, y/s, z/s}; }
    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3& operator*=(int s) { x*=s; y*=s; z*=s; return *this; }
    Vec3& operator/=(int s) { x/=s; y/=s; z/=s; return *this; }
};


struct Node { // Node class to cleanly group data
    int mass;
    Vec3 com; // center of mass
    Vec3 center; // center of the cube
    int half_size;

    int children[8]; // Array of int with size 8 (indices of the 8 children)
    int particle; // Index of the particle, -1 if it's not a leaf
};


bool has_children(Node& node) {
    return node.children[0] != -1;
}


int get_octant(Node& node, int p_idx) {
    Vec3& particle = particles[p_idx];

    Vec3 rel_cords = particle - node.center; // Moves coordinate system to node center

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

    // If for some reson no octant gets selected
    return -1; // Will definitely lead to an error (intentionally)
}


// This is the least clean code I've written in a while, please don't judge
void create_children(Node& node) {
    Node child_0, child_1, child_2, child_3, child_4, child_5, child_6, child_7;

    child_0.center = node.center;
    child_0.half_size = node.half_size / 2;
    child_0.particle = -1;
    fill(begin(child_0.children), end(child_0.children), -1);
    child_0.center.x += child_0.half_size;
    child_0.center.y += child_0.half_size;
    child_0.center.z += child_0.half_size;
    

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
    node.children[0] = nodes.size() - 1;

    nodes.push_back(child_1);
    node.children[1] = nodes.size() - 1;

    nodes.push_back(child_2);
    node.children[2] = nodes.size() - 1;

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


void insert(int node_idx, int p_idx) {
    // Insert particle p_idx into node node_idx
    Node& node = nodes[node_idx]; // Only store reference, avoid copying

    // Empty leaf
    if (node.particle == -1 and not has_children(node)) { 
        node.particle = p_idx;
        return;
    }

    // Internal node
    if (has_children(node)) { 
        int oct = get_octant(node, p_idx);
        insert(node.children[oct], p_idx); // Insert node into correct child
        return;
    }

    // Leaf with particle
    
    int old_p_idx = node.particle;
    node.particle = -1;

    create_children(node);

    insert(node_idx, old_p_idx);

    insert(node_idx, old_p_idx);
}

void compute_mass_and_com(int node_idx) {
    Node& node = nodes[node_idx];

    if (node.particle = -1) { // leaf
        node.mass = m[node.particle];
        node.com = particles[node.particle];
        return;
    }

    // Inside node
    node.mass = 0;
    node.com.x = 0;
    node.com.y = 0;
    node.com.z = 0;

    for (int i = 0; i < 8; i++) {
        compute_mass_and_com(node.children[i]); // Update children first

        Node& child = nodes[node.children[i]];
        node.mass += child.mass; // Add child mass to total mass
        node.com += child.com * child.mass; // Shift center of gravity
    }

    node.com /= node.mass; // Fix the scaling caused by the mass (basically a weighted average)
}
