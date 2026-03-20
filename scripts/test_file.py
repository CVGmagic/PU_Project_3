from simulation import random_shape_creator_2D, random_shape_creator_3D, initial_conditions
from renderers import renderer_2D, renderer_3D
from acceleration import acceleration_calculator_3D
import numpy as np
import matplotlib.pyplot as plt


m = initial_conditions.mass
r = random_shape_creator_3D.create_sphere_3D(np.array([0, 0, 0]), 1, 1000)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
renderer_3D.plot_points_3D(r, ax)
plt.show()
#TODO: explain Maïc the 3 lines above
acceleration = acceleration_calculator_3D.create_array_acc(r)
# it works!

print(r)
print(acceleration)
