from vispy import scene, app
from acceleration.acceleration_calculator_3D import calculate_attractive_acceleration
import numpy as np
from renderers import renderer_3D

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Particle data
r = np.array([[0, 0, 0], [15, 0, 0]])
sizes = np.array([10, 1])
m = np.array([0, 1])

# Create markers (GPU points)
scatter = scene.visuals.Markers()
renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)
view.add(scatter)

dt = 0.01
v = np.zeros((r.shape[0], 3))
v[1][1] = 2
a = calculate_attractive_acceleration(r, m)
v += a * dt / 2

def update(event):
    # update positions every frame
    global r, v
    r += v * dt

    a = calculate_attractive_acceleration(r, 100)

    v += a * dt

    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)


timer = app.Timer(0.016, connect=update, start=True)  # ~60 FPS
app.run()

"""
VERY IMPORTANT !!!!!!!!
My computer almost crashed after running the program with just 
100 particles without this line
"""
canvas.close()
