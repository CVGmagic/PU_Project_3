import matplotlib.pyplot as plt
import numpy as np

def snapshot_2D(r: np.ndarray):
    x = r[:, 0]
    y = r[:, 1]
    plt.plot(x, y)
    plt.show()