from simulation import random_shape_creator_2D, random_shape_creator_3D, initial_conditions
from renderers import renderer_2D, renderer_3D
import numpy as np
import matplotlib.pyplot as plt
from acceleration.acceleration_calculator_3D import calc_acc_rep_np, calculate_gravitational_acceleration
from simulation.constants import G, eps_sq
from vispy import scene, app
from simulation.energy_calculator import calculate_potential_energies

n = initial_conditions.n


""" Try same with PyVis"""
def create_canvas():
    canvas = scene.SceneCanvas(keys='interactive', show=True) # creates a window
    view = canvas.central_widget.add_view() # adds a scene to window
    view.camera = 'turntable' # you can change perspective in your scene

    # Particle data
    r = random_shape_creator_3D.create_sphere_3D(np.array([0, 0, 0]), 1, n) # creates the random points
    sizes = np.random.rand(n) * 20 # saves a list with n-elements which all have different sizes

    # Create markers (GPU points)
    scatter = scene.visuals.Markers() # an empty list (kinda)
    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes) # fills scatter with coordinates + sizes
    view.add(scatter) # adds scatter (basically points) to view
    return canvas, scatter, sizes, r

canvas, scatter, sizes, r = create_canvas()
""" add n -> n+1 and add line 57 when we add the sum but also add the coordinate of the sun that it works
len(m) == len(r)"""
dt = 0.0001
m = np.full(n, 100) # creates array with n elements and (masses of 100)
v = np.zeros((n, 3)) # v has n elements in 3D filled with 0's
a = calc_acc_rep_np(r, m) # calculates the acceleration of every single r based on their location (r)
#m[n] = 33300000
v += a * dt / 2 # updates v

step_count = 1
max_steps = 100



def update_conditions_rep(): #  'event' is needed with the timer which later allows the command timer.stop()
    global r, v, m, dt

    a = calc_acc_rep_np(r, m)
    v = a * dt

    return


def update_conditions(): #  'event' is needed with the timer which later allows the command timer.stop()
    global r, v, m, dt

    sum_acc_gravity, sum_acc_pressure = calculate_potential_energies(r, m)

    energy_relation = sum_acc_gravity / sum_acc_pressure

    a = calculate_gravitational_acceleration(r, m, energy_relation)
    v = a * dt
    return a, v, energy_relation


def update_starting_position(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with wrong gravity
    global r, v, m, dt, max_steps, step_count

    a = calc_acc_rep_np(r, m)
    a += -0.5 * r # prevents the sphere from exploding

    v += a * dt
    v *= 0.9 # adds damping

    r += v * dt

    renderer_3D.plot_points_3D_PyVis(r,scatter, sizes)

    if step_count == max_steps:

        timer1.stop()  # stop calling update
        update_conditions()
        timer_accurate.start()

    step_count += 1


def update_simulation(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with correct gravity
    global r, v, m, dt, energy_relation

    r += v * dt

    a = calculate_gravitational_acceleration(r, m, energy_relation)

    v += a * dt

    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)


timer1 = app.Timer(0.016, connect=update_starting_position, start=True)  # ~60 FPS but actually limited by calculations so same as while run do
timer_accurate = app.Timer(0.016, connect=update_simulation, start=False) # ~60 FPS but actually limited by calculations so same as while run do
timer_barneshut =  app.Timer(0.016, connect=update_simulation, start=False)
app.run() # starts the simulation



"""
VERY IMPORTANT !!!!!!!!
My computer almost crashed after running the program with just 
100 particles without this line
"""
canvas.close() # give memory free as soon as window gets closed