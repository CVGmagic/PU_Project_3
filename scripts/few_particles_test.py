from measure_tools.measure_functions import plot_elongations

# Etwa ab 0.773565 verringert sich die elongation zunächst, weil der planet noch kreisförmiger wird
distances = [0.873566] # , 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

plot_elongations(distances, timesteps=100, pre_relax=0)

