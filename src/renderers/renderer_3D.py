import matplotlib.pyplot as plt
import numpy as np

def plot_points_3D(r: np.ndarray, ax):
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    ax.scatter(x, y, z, linestyle="", marker="o")

