import matplotlib.pyplot as plt
import numpy as np

def plot_points_2D(r: np.ndarray, ax):
    x = r[:, 0]
    y = r[:, 1]
    ax.plot(x, y, linestyle="", marker="o")
