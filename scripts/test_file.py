from simulation import random_shape_creator_2D, random_shape_creator_3D, initial_conditions
from renderers import renderer_2D, renderer_3D
from acceleration import acceleration_calculator_3D
import numpy as np
import matplotlib.pyplot as plt
from acceleration.acceleration_calculator_3D import calc_acc_rep_np
from vispy import scene, app

n = initial_conditions.n
m = initial_conditions.mass

"""
r = random_shape_creator_3D.create_cuboid_3D(np.array([0, 0, 0]), np.array([1, 1, 1]), n)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_aspect("equal")
renderer_3D.plot_points_3D_plt(r, ax)
plt.pause(1e-30)

dt = 0.01
v = np.zeros((n, 3))
a = calc_acc_rep_np(r, 100)
v += a * dt / 2

for i in range(30):
    r += v * dt

    a = calc_acc_rep_np(r, 100)

    v += a * dt
    ax.clear()
    renderer_3D.plot_points_3D(r, ax)
    plt.pause(1e-30)

acceleration = acceleration_calculator_3D.create_array_acc(r, m)
# it works!
"""

""" Try same with PyVis"""
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Particle data
r = random_shape_creator_3D.create_cuboid_3D(np.array([0, 0, 0]), np.array([1, 1, 1]), n)
sizes = np.random.rand(10000) * 5  # radius

# Create markers (GPU points)
scatter = scene.visuals.Markers()
renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)
view.add(scatter)

dt = 0.01
v = np.zeros((n, 3))
a = calc_acc_rep_np(r, 100)
v += a * dt / 2

def update(event):
    # update positions every frame
    r += v * dt

    a = calc_acc_rep_np(r, 100)

    v += a * dt

    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)


timer = app.Timer(0.016, connect=update, start=True)  # ~60 FPS
app.run()

