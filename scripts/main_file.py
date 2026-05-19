import numpy as np
import time
from vispy import app, scene
from simulation import random_shape_creator_3D
from acceleration.acceleration_calculator_3D import calc_acc_rep_np, calculate_gravitational_acceleration
from simulation.energy_calculator import calculate_potential_energies, calculate_new_energy_relation, calculate_energy_relation
from renderers import renderer_3D
from acceleration import barnes_hut_python
from measure_tools.measure_functions import compute_principal_axis_lengths, \
    compute_principal_axis_lengths_outliers_excluded, planet_com, circular_orbit_velocity, roche_limit, compute_elongation
import matplotlib.pyplot as plt
import math
from simulation.constants import G


def update_starting_position(event): # 'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with wrong gravity
    global r, v, m, max_steps, step_count, barnes_hut

    dt = 0.0001

    a = calc_acc_rep_np(r, np.full_like(m, 100))

    v += a * dt
    v *= 0.9 # adds damping

    r += v * dt

    renderer_3D.plot_points_3D_PyVis(r,scatter, sizes)

    if step_count == max_steps:

        timer1.stop()  # stop calling update

        scale_r_back()
        update_conditions()
        add_solar_point(distance_star, v_rotation, mass_star)

        """Half step to prepare for Leapfrog"""
        a = calculate_gravitational_acceleration(r, m, energy_relation)
        v += a * (dt / 2)

        if not barnes_hut:
            timer_accurate.start()
        if barnes_hut:
            timer_barnes_hut.start()

    step_count += 1


def scale_r_back():
    global r

    max_dist = np.sqrt(np.sum(r * r, axis=1).max())
    r = r / max_dist


def update_conditions(): #  'event' is needed with the timer which later allows the command timer.stop()
    global r, v, m, dt, energy_relation

    #energy_relation = calculate_energy_relation(r, m)
    energy_relation = calculate_new_energy_relation(r.shape[0], m[-1], 1)
    #print(energy_relation)


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
    global r, v, m, dt, energy_relation, view, total_time, sim_step_count, sim_start_time, elongations, n, mass_planet, mass_star, scatter

    r += v * dt
    a = calculate_gravitational_acceleration(r, m, energy_relation)
    v += a * dt


    view.camera.center = planet_com(r, m)
    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)

    elongations.append(compute_elongation(r, percentile=95))

    sim_step_count += 1

    if sim_step_count == plot_steps:
        timer_accurate.stop()

        colors = np.full((n+1, 4), (1, 1, 1, 1))  # All set to white at beginning

        com = np.mean(r[1:], axis=0)
        dists = np.linalg.norm(r - com, axis=1)

        threshold = np.percentile(dists[1:], 95)
        mask = dists > threshold

        colors[mask] = (1, 0, 0, 1)
        colors[0] = (1, 1, 1, 1)

        renderer_3D.plot_points_3D_PyVis(r, scatter, sizes, colors)  # fills scatter with coordinates + sizes

        timer_outliers.start()

    if False:
        plt.figure()
        t = np.linspace(0, plot_steps * dt, plot_steps)
        plt.plot(t, elongations)
        plt.title(f"Elongation vs Time \n n={n}, dt={dt}, m_planet={mass_planet}, m_star={mass_star}")
        plt.xlabel("Time [s]")
        plt.ylabel("Elongation")
        plt.show()


def update_simulation_barnes_hut(event): #  'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with correct gravity
    global r, v, m, dt, energy_relation, view, total_time, sim_step_count, sim_start_time

    a = barnes_hut_python.compute_accelerations(r, m, energy_relation)
    v += a * dt
    r += v * dt

    view.camera.center = planet_com(r, m)
    renderer_3D.plot_points_3D_PyVis(r, scatter, sizes)


def rotate_camera(event):
    view.camera.azimuth += 0.25


elongations = []
plot_steps = 3000

n = 500
barnes_hut = False
dt = 0.001
mass_planet = 5e10
mass = mass_planet / n
max_steps = 50
mass_star = mass_planet * 1e11 # Made mass dependent on planet mass instead of particle mass
roche = roche_limit(mass_planet, mass_star, 1)
distance_star =  1 * roche # 3.3 (gives a nice spiral)
v_rotation = circular_orbit_velocity(distance_star, mass_star)
stop_time = False

def create_canvas():
    global canvas, scatter, sizes, view
    canvas = scene.SceneCanvas(keys='interactive', show=True) # creates a window
    view = canvas.central_widget.add_view() # adds a scene to window
    view.camera = 'turntable' # you can change perspective in your scene

    # Particle data
    random_points = random_shape_creator_3D.create_sphere_3D(np.array([0, 0, 0]), 1, n) # creates the random points
    sizes = np.full(n, 10)

    # Create markers (GPU points)
    scatter = scene.visuals.Markers() # an empty list (kinda)
    renderer_3D.plot_points_3D_PyVis(random_points, scatter, sizes) # fills scatter with coordinates + sizes
    view.add(scatter) # adds scatter (basically points) to view
    return random_points

r = create_canvas()

m = np.full(n, mass) # creates array with n elements and (masses of 100)
v = np.zeros((n, 3)) # v has n elements in 3D filled with 0's
a = calc_acc_rep_np(r, np.full_like(m, 100)) # calculates the acceleration of every single r based on their location (r)
#m[n] = 33300000
v += a * dt / 2 # updates v
sim_step_count = 0

step_count = 0

timer1 = app.Timer(0.0016, connect=update_starting_position, start=True)  # ~60 FPS but actually limited by calculations so same as while run do
timer_accurate = app.Timer(0.0016, connect=update_simulation, start=False) # ~60 FPS but actually limited by calculations so same as while run do
timer_barnes_hut =  app.Timer(0.0016, connect=update_simulation_barnes_hut, start=False) # not correct function
timer_outliers = app.Timer(0.0016, connect=rotate_camera, start=False)

app.run()