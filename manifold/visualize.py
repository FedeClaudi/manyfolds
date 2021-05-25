import matplotlib.pyplot as plt

# from myterial import blue_grey, grey_dark, grey

blue = "#c3c3db"


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
    ax.zaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    return ax
