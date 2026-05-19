from renderers.renderer_3D import plot_points_3D_PyVis
import numpy as np
from vispy import app, scene
from simulation.random_shape_creator_3D import single_point_cuboid
import time


n = 200
r = 1
m = np.array([0, 0, 0])

canvas = scene.SceneCanvas(keys='interactive', show=True) # creates a window
view = canvas.central_widget.add_view() # adds a scene to window
view.camera = 'turntable' # you can change perspective in your scene

# Particle data
points = np.zeros((n, 3))
sizes = np.zeros(n)
colors = np.full((n, 4), (1, 0, 0, 1)) # All set to red at beginning

# Create markers (GPU points)
scatter = scene.visuals.Markers() # an empty list (kinda)
plot_points_3D_PyVis(points, scatter, sizes, colors) # fills scatter with coordinates + sizes
view.add(scatter) # adds scatter (basically points) to view


lower = m-r
upper = m+r

canvas.show()
time.sleep(1)
start_time = time.perf_counter()

for i in range(n):
    point = single_point_cuboid(lower, upper)
    view.camera.azimuth += 0.5

    sizes[i] = 10
    while np.linalg.norm(point) > r:
        points[i] = point
        plot_points_3D_PyVis(points, scatter, sizes, colors)
        # Redraw
        canvas.update()
        canvas.app.process_events()

        point = single_point_cuboid(lower, upper)
        view.camera.azimuth += 0.5

    points[i] = point
    colors[i] = (1, 1, 1, 1)
    plot_points_3D_PyVis(points, scatter, sizes, colors)

    # Redraw
    canvas.update()
    canvas.app.process_events()

end_time = time.perf_counter()
print(end_time - start_time)

while True:
    view.camera.azimuth += 0.5
    canvas.update()
    canvas.app.process_events()
