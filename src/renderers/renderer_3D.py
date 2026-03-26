import matplotlib.pyplot as plt
import numpy as np
from vispy import scene, app

def plot_points_3D_plt(r: np.ndarray, ax) -> None:
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    ax.scatter(x, y, z, linestyle="", marker="o")

    return


def plot_points_3D_PyVis(r: np.ndarray, scatter, sizes=np.array([])) -> None:
    """
    Creates a 3D plot on scatter using the positions in r, and the sizes in sizes.
    If no sizes are provided, 1 is used as a default size
    """
    if sizes.size() == 0:
        sizes = np.ones((r.size()))
    elif sizes.size() != r.size():
        raise ValueError("r and sizes must have the same length")

    scatter.set_data(positions, size=sizes, face_color=(1, 1, 1, 1))

    return
