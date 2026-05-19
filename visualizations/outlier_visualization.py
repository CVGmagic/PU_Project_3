import numpy as np
from renderers.renderer_3D import plot_points_3D_PyVis
import numpy as np
from vispy import app, scene
from simulation.random_shape_creator_3D import create_sphere_3D
import time

n = 500
m = np.array([0, 0, 0])
r = 1

canvas = scene.SceneCanvas(keys='interactive', show=True) # creates a window
view = canvas.central_widget.add_view() # adds a scene to window
view.camera = 'turntable' # you can change perspective in your scene

# Particle data
points = create_sphere_3D(m, r, n)
points[:, 2] *= 0.8
points[:, 1] *= 2
sizes = np.full(n, 10)
colors = np.full((n, 4), (1, 1, 1, 1)) # All set to white at beginning

com = np.mean(points, axis=0)
dists = np.linalg.norm(points - com, axis=1)

threshold = np.percentile(dists, 95)
mask = dists > threshold

colors[mask] = (1, 0, 0, 1)

# Create markers (GPU points)
scatter = scene.visuals.Markers() # an empty list (kinda)
plot_points_3D_PyVis(points, scatter, sizes, colors) # fills scatter with coordinates + sizes
view.add(scatter) # adds scatter (basically points) to view


def update(event):
    view.camera.azimuth += 0.25

timer = app.Timer(0.0016, connect=update, start=True)

app.run()




