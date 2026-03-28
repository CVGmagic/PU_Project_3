from simulation import random_shape_creator_2D, random_shape_creator_3D, initial_conditions
from renderers import renderer_2D, renderer_3D
import numpy as np
import matplotlib.pyplot as plt
from acceleration.acceleration_calculator_3D import calc_acc_rep_np, calculate_complete_acceleration, create_array_acc
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
canvas = scene.SceneCanvas(keys='interactive', show=True) # creates a window
view = canvas.central_widget.add_view() # adds a scene to window
view.camera = 'turntable' # you can change perspective in your scene

# Particle data
r = random_shape_creator_3D.create_cuboid_3D(np.array([0, 0, 0]), np.array([1, 1, 1]), n) # creates the random points
sizes = np.random.rand(n) * 10 # saves a list with n-elements which all have different sizes

# Create markers (GPU points)
scatter = scene.visuals.Markers() # an empty list (kinda)
renderer_3D.plot_points_3D_PyVis(r, scatter, sizes) # fills scatter with coordinates + sizes
view.add(scatter) # adds scatter (basically points) to view

""" add n -> n+1 and add line 57 when we add the sum but also add the coordinate of the sun that it works
len(m) == len(r)"""
dt = 0.0001
m = np.full(n, 100) # creates array with n elements and (masses of 100)
v = np.zeros((n, 3)) # v has n elements in 3D filled with 0's
a = calc_acc_rep_np(r, m, 0) # calculates the acceleration of every single r based on their location (r)
#m[n] = 33300000
v += a * dt / 2 # updates v

step_count = 1
max_steps = 100

power_coefficient = 0

def update_conditions(): #  'event' is needed with the timer which later allows the command timer.stop()
    global r, v, m, dt

    sum_acc_gravity = 0
    sum_acc_pressure = 0

    #for i in range(n):

    a = calculate_complete_acceleration(r, m)
    v = a * dt

    return

def update_starting_position(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with wrong gravity
    global r, v, m, dt, max_steps, step_count, power_coefficient

    r += v * dt

    a = calc_acc_rep_np(r, m, power_coefficient)

    v += a * dt

    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)

    if step_count % max_steps == 0:

        update_conditions()

        if step_count == 2 * max_steps:

            update_conditions()
            timer1.stop()  # stop calling update
            timer2.start() # start new function

        step_count += 1
        power_coefficient += 1
        return

    step_count += 1

def update_simulation(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with correct gravity
    global r, v, m, dt

    r += v * dt

    a = calculate_complete_acceleration(r, m)

    v += a * dt

    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)

timer1 = app.Timer(0.016, connect=update_starting_position, start=True)  # ~60 FPS but actually limited by calculations so same as while run do
timer2 = app.Timer(0.016, connect=update_simulation, start=False) # ~60 FPS but actually limited by calculations so same as while run do
app.run() # starts the simulation



"""
VERY IMPORTANT !!!!!!!!
My computer almost crashed after running the program with just 
100 particles without this line
"""
canvas.close() # give memory free as soon as window gets closed