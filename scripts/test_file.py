from simulation import random_shape_creator_2D, random_shape_creator_3D
from renderers import renderer_2D, renderer_3D
import numpy as np
import matplotlib.pyplot as plt


r = random_shape_creator_3D.create_sphere_3D(np.array([0, 0, 0]), 1, 1000)
print(r)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
renderer_3D.plot_points_3D(r, ax)
plt.show()