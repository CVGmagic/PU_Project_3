from vispy import app, scene
import numpy as np
from simulation import random_shape_creator_3D
from acceleration.acceleration_calculator_3D import calculate_gravitational_acceleration, calc_acc_rep_np
from simulation.energy_calculator import calculate_potential_energies

def update_starting_position(r: np.ndarray, v: np.ndarray, m: np.ndarray, dt=0.001, steps=50, barnes_hut=False): # 'event' is needed with the timer which later allows the command timer.stop()
    # update positions every frame with wrong gravity

    for _ in range(steps):
        a = calc_acc_rep_np(r, m)

        v += a * dt
        v *= 0.9 # adds damping

        r += v * dt

        renderer_3D.plot_points_3D_PyVis(r,scatter, sizes)

    scale_r_back(r)
    update_conditions()
    add_solar_point(distance_star, v_rotation, mass_star)


def scale_r_back(r: np.ndarray, radius: float=1) -> None:
    """Scales all point in r such that the maximum point is inside a given radius"""
    max_dist = np.sqrt(np.sum(r * r, axis=1).max())
    # TODO Maybe switch to some amount of standard deviations
    r *= radius / max_dist
    return


def update_conditions(r: np.ndarray, v: np.ndarray, m: np.ndarray, dt: float, energy_relation: float) -> None:
    """Updates r using the specified conditions"""
    sum_acc_gravity, sum_acc_pressure = calculate_potential_energies(r, m)

    energy_relation = sum_acc_gravity / sum_acc_pressure

    a = calculate_gravitational_acceleration(r, m, energy_relation)
    v = a * dt
    return


def add_solar_point(r, m, v, distance_star: float, v_rotation: float, central_body_m: float) -> None:
    """Adds a sun with the specified parameters to r"""
    v = np.full((n, 3), [0.0, float(v_rotation), 0.0], dtype=np.float64)
    central_body_v = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    v = np.vstack((central_body_v, v))

    central_body_r = [distance_star, 0, 0]
    r = np.vstack((central_body_r, r))

    m = np.concatenate(([central_body_m], m))
    return r, m, v


def create_canvas(canvas, scatter, sizes, view) -> np.ndarray:
    """Fills in the canvas and returns an array of random spherical points"""
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


def generate_points(particle_count, planet_radius, star_distance, dt=0.001, steps=50, pre_relax: int=0, m: np.ndarray=None) -> np.ndarray:
    r = random_points = random_shape_creator_3D.create_sphere_3D(np.array([0, 0, 0]), planet_radius, particle_count)
    v = np.zeros_like(r)

    if m is None and pre_relax == 0:
        m = np.full(particle_count, 100)
    elif m is None and pre_relax != 0:
        raise ValueError("If you want to pre-relax the planet, you must give the masses of the particles")

    for _ in range(steps):
        a = calc_acc_rep_np(r, m)

        v += a * dt
        v *= 0.9  # adds damping

        r += v * dt

    scale_r_back(r, planet_radius)

    if pre_relax != 0:
        sum_acc_gravity, sum_acc_pressure = calculate_potential_energies(r, m)
        energy_relation = sum_acc_gravity / sum_acc_pressure

        a = calculate_gravitational_acceleration(r, m, energy_relation)
        v = a * (dt / 2)

        for i in range(pre_relax):
            r += v * dt
            a = calculate_gravitational_acceleration(r, m, energy_relation)
            v += a * dt


    r = np.vstack(([star_distance, 0, 0], r))

    return r



