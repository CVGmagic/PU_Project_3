from measure_tools.measure_functions import benchmark_computation

particle_counts = [10 * 2**i for i in range(10)]
opening_angles = [0.4]

benchmark_computation(particle_counts, opening_angles=opening_angles, trials=100)