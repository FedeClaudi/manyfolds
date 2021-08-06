import matplotlib.pyplot as plt
import numpy as np
from vedo.colors import hex2rgb, rgb2hex


blue = "#c3c3db"
blue_dark = "#a9a9cc"


def make_palette(c1, c2, N):
    c1 = np.array(hex2rgb(c1))
    c2 = np.array(hex2rgb(c2))
    cols = []
    for f in np.linspace(0, 1, N, endpoint=True):
        c = c1 * (1 - f) + c2 * f
        cols.append(rgb2hex(c))
    return cols


def make_3D_ax(nolim=True):
    plt.figure(figsize=(9, 9))
    ax = plt.axes(projection="3d")
    ax.set(
        xlabel="$x^1$", ylabel="$x^2$", zlabel="$x^3$",
    )

    if not nolim:
        ax.set(
            xlim=[-1, 1],
            ylim=[-1, 1],
            zlim=[-1, 1],
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1],
            zticks=[-1, 0, 1],
        )

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    return ax
