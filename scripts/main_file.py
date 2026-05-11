import numpy as np
from vispy import app
from test_file import create_canvas, update_starting_position, update_conditions, update_simulation
from acceleration.acceleration_calculator_3D import calc_acc_rep_np


canvas, scatter, sizes, r = create_canvas()
barnes_hut = False
simulation_start = False
n = 500
dt = 0.0001
m = np.full(n, 100) # creates array with n elements and (masses of 100)
v = np.zeros((n, 3)) # v has n elements in 3D filled with 0's
a = calc_acc_rep_np(r, m) # calculates the acceleration of every single r based on their location (r)
#m[n] = 33300000
v += a * dt / 2 # updates v

step_count = 0
max_steps = 50

timer1 = app.Timer(0.016, connect=update_starting_position, start=True)  # ~60 FPS but actually limited by calculations so same as while run do
timer_accurate = app.Timer(0.016, connect=update_simulation, start=False) # ~60 FPS but actually limited by calculations so same as while run do

a, v, energy_relation = update_conditions()

# add central point

if not barnes_hut:
    timer_accurate.start()


app.run()