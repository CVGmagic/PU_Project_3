import numpy as np
from vispy import app, scene
from simulation import random_shape_creator_3D
from acceleration.acceleration_calculator_3D import calc_acc_rep_np, calculate_gravitational_acceleration
from simulation.energy_calculator import calculate_potential_energies
from renderers import renderer_3D
from acceleration import barnes_hut_python


def update_starting_position(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with wrong gravity
    global r, v, m, dt, max_steps, step_count, barnes_hut

    a = calc_acc_rep_np(r, m)

    v += a * dt
    v *= 0.9 # adds damping

    r += v * dt

    renderer_3D.plot_points_3D_PyVis(r,scatter, sizes)

    if step_count == max_steps:

        timer1.stop()  # stop calling update

        scale_r_back()
        update_conditions()
        add_solar_point(distance_star, v_rotation, mass_star)

        if not barnes_hut:
            timer_accurate.start()
        if barnes_hut:
            timer_barnes_hut.start()

    step_count += 1


def scale_r_back():
    global r

    max_dist = np.sqrt(np.sum(r * r, axis=1).max())
    r = r/max_dist

def update_conditions(): #  'event' is needed with the timer which later allows the command timer.stop()
    global r, v, m, dt, energy_relation

    sum_acc_gravity, sum_acc_pressure = calculate_potential_energies(r, m)

    energy_relation = sum_acc_gravity / sum_acc_pressure

    a = calculate_gravitational_acceleration(r, m, energy_relation)
    v = a * dt

def add_solar_point(distance_star, v_rotation, central_body_m):
    global r, m, v, sizes

    v = np.full((n, 3), [0.0, float(v_rotation), 0.0], dtype=np.float64)
    central_body_v = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    v = np.vstack((central_body_v, v))

    central_body_r = [distance_star, 0, 0]
    r = np.vstack((central_body_r, r))

    m = np.concatenate(([central_body_m], m))

    central_body_size = 50
    sizes = np.insert(sizes, 0, central_body_size)


def update_simulation(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with correct gravity
    global r, v, m, dt, energy_relation, view

    r += v * dt

    a = calculate_gravitational_acceleration(r, m, energy_relation)

    v += a * dt

    view.camera.center = r[0]
    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)

def update_simulation_barnes_hut(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with correct gravity
    global r, v, m, dt, energy_relation, view

    r += v * dt

    a = barnes_hut_python.compute_accelerations(r, m)

    v += a * dt

    view.camera.center = r[0]
    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)

n = 500
barnes_hut = False
dt = 0.0001
mass = 100
max_steps = 50
v_rotation = 400
distance_star = 4
mass_star = mass * 5000

def create_canvas():
    global canvas, scatter, sizes, view
    canvas = scene.SceneCanvas(keys='interactive', show=True) # creates a window
    view = canvas.central_widget.add_view() # adds a scene to window
    view.camera = 'turntable' # you can change perspective in your scene

    # Particle data
    random_points = random_shape_creator_3D.create_sphere_3D(np.array([0, 0, 0]), 1, n) # creates the random points
    sizes = np.random.rand(n) * 15 # saves a list with n-elements which all have different sizes

    # Create markers (GPU points)
    scatter = scene.visuals.Markers() # an empty list (kinda)
    renderer_3D.plot_points_3D_PyVis(random_points, scatter, sizes) # fills scatter with coordinates + sizes
    view.add(scatter) # adds scatter (basically points) to view
    return random_points

r = create_canvas()

m = np.full(n, mass) # creates array with n elements and (masses of 100)
v = np.zeros((n, 3)) # v has n elements in 3D filled with 0's
a = calc_acc_rep_np(r, m) # calculates the acceleration of every single r based on their location (r)
#m[n] = 33300000
v += a * dt / 2 # updates v

step_count = 0

timer1 = app.Timer(0.0016, connect=update_starting_position, start=True)  # ~60 FPS but actually limited by calculations so same as while run do
timer_accurate = app.Timer(0.0016, connect=update_simulation, start=False) # ~60 FPS but actually limited by calculations so same as while run do
timer_barnes_hut =  app.Timer(0.0016, connect=update_simulation_barnes_hut, start=False) # not correct function

app.run()