from measure_tools.measure_functions import plot_elongations, benchmark_computation

"""
opening_angles = [0.8]
particle_counts = [100, 250, 400, 750, 1250, 2500, 5000, 10_000, 15_000]
benchmark_computation(particle_counts, True, True, opening_angles, trials=10)
"""

# Etwa ab 0.773565 verringert sich die elongation zunächst, weil der planet noch kreisförmiger wird
distances = [2] # , 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

plot_elongations(distances, timesteps=10000, pre_relax=1000, m_planet=5e10)

