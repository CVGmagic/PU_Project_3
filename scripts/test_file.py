from simulation import random_shape_creator_2D
from renderers import renderer_2D
import numpy as np
import matplotlib.pyplot as plt


r = random_shape_creator_2D.create_circle_2D(np.array([0, 0]), 1, 1000)
print(r)
fig, ax = plt.subplots()
renderer_2D.plot_points_2D(r, ax)
plt.show()