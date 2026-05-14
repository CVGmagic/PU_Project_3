from measure_tools.measure_functions import plot_elongations, benchmark_computation


opening_angles = [0.4]
particle_counts = [300, 400, 500, 600, 700, 800, 900]
benchmark_computation(particle_counts, True, True, opening_angles, trials=10)

# Etwa ab 0.773565 verringert sich die elongation zunächst, weil der planet noch kreisförmiger wird
distances = [0.873566] # , 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

plot_elongations(distances, timesteps=100, pre_relax=0)

