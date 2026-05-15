from measure_tools.measure_functions import plot_elongations, benchmark_computation

"""
opening_angles = [0.8]
particle_counts = [100, 250, 400, 750, 1250, 2500, 5000, 10_000, 15_000]
benchmark_computation(particle_counts, True, True, opening_angles, trials=10)
"""

# Etwa ab 0.773565 verringert sich die elongation zunächst, weil der planet noch kreisförmiger wird
distances = [1, 1.2, 1.4, 1.6, 1.8, 2] # , 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

plot_elongations(distances, timesteps=1000, pre_relax=1000)

