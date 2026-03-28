from vispy import scene, app
from acceleration.acceleration_calculator_3D import calculate_attractive_acceleration
import numpy as np
from renderers import renderer_3D
from simulation.energy_calculator import get_total_energy
from simulation.random_shape_creator_3D import create_relaxed_sphere_3D

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

n = 10
# Particle data
r = np.zeros((n + 1, 3))
r[1:] = create_relaxed_sphere_3D(np.array([100, 0, 0]), 3, n)

print(r)
sizes = np.full(n + 1, 10)
sizes[0] = 30
m = np.ones(n + 1)
m[0] = 100000

# Create markers (GPU points)
scatter = scene.visuals.Markers()
renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)
view.add(scatter)
view.camera.set_range()

dt = 0.01
v = np.zeros((r.shape[0], 3))
v[1:] = [0, 50, 0]
a = calculate_attractive_acceleration(r, m)
v += a * dt / 2

def update(event):
    # update positions every frame
    global r, v
    r += v * dt

    """ Print distance for debugging"""
    dist = r[0] - r[1]
    print(get_total_energy(r, m, v))

    a = calculate_attractive_acceleration(r, m)

    v += a * dt

    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)


timer = app.Timer(1e-32, connect=update, start=True)  # ~60 FPS
app.run()

"""
VERY IMPORTANT !!!!!!!!
My computer almost crashed after running the program with just 
100 particles without this line
"""
canvas.close()
