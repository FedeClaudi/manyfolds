import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

from fcutils.progress import track
from fcutils.plot.figure import clean_axes
from myterial import salmon

from manifold.manifolds._embeddings import parse2D
from manifold.manifolds.embeddings import Embedding
from manifold import Plane
from manifold.tangent_vector import get_tangent_vector

MANIFOLD_COLOR = "#b8b6d1"
POINT_COLOR = "#3838BA"


@parse2D
def _plane_2d(p0, p1):
    return (p0, p1, 0)


# @parse2D
# def _curvy_plane_3d(p0, p1):
#     return (p0, p1, np.sin(2 * np.pi * p0) * np.sin(2 * np.pi * p1) * .3 + 0.3)


@parse2D
def _curvy_plane_3d(p0, p1):
    return (
        p0,
        p1,
        0.3 - np.sin(1.5 * np.pi * p0) * np.sin(1.5 * np.pi * p1) * 0.3,
    )


def vifield_one(p):
    return (np.sin(2 * np.pi * p[0]), np.sin(2 * np.pi * p[1]))


def vifield_two(p):
    return (1 - np.sin(2 * np.pi * p[0]), 1 - np.sin(2 * np.pi * p[1]))


def vfield_three(p):
    return (np.sin(2 * np.pi * (p[0] + 0.2)), np.sin(2 * np.pi * (p[0] + 0.2)))


plane_2d = Embedding("plane 2d", _plane_2d)
plane_3d_embedding = Embedding("plane 3d", _curvy_plane_3d)


class PyplotVisualizer:
    """
        Visualize 3D manifolds and tangent vectors using matplotlib.
        Currently only works with plane manifold. It shows the plane in 2D
        with tangent vectors on it and in 3D in embedding space.
    """

    tvecs_scale = [0.03, 0.04]

    def __init__(self):
        self.make_figure()

        self.M2d = Plane(plane_2d, n_sample_points=12)
        self.M3d = Plane(plane_3d_embedding, n_sample_points=12)

        X = np.arange(0, 1.02, 0.02)
        Y = np.arange(0, 1.02, 0.02)
        self._X, self._Y = np.meshgrid(X, Y)

        self.draw_plane_2d()
        self.draw_plane_3d()
        self.draw_tangent_vectors((vifield_one, vifield_one))

    def make_figure(self):
        self.f = plt.figure(figsize=(14, 9))

        self.ax2d = plt.subplot(121, projection="3d")
        self.ax3d = plt.subplot(122, projection="3d")
        clean_axes(self.f)
        self.f.tight_layout()

        self.ax2d.set(xlim=[0, 1], ylim=[0, 1])
        self.ax2d.view_init(elev=90, azim=0)
        self.ax2d.axis("off")

        self.ax3d.set(
            xlim=[0, 1],
            ylim=[0, 1],
            zlim=[0, 1],
            xticks=[0, 1],
            yticks=[0, 1],
            zticks=[0, 1],
        )
        self.ax3d.view_init(elev=15, azim=-85)
        self.ax3d.axis("off")

    def draw_plane_2d(self):
        self.ax2d.plot_surface(
            self._X,
            self._Y,
            np.zeros_like(self._X),
            color=MANIFOLD_COLOR,
            shade=False,
            alpha=1,
        )

    def draw_plane_3d(self):
        k = self._X.shape[0]
        points = self.M3d.sample(k - 1, fill=True)
        points_coordinates = np.vstack([pt.embedded for pt in points])

        X = points_coordinates[:, 0].reshape(-1, k)
        Y = points_coordinates[:, 1].reshape(-1, k)
        Z = points_coordinates[:, 2].reshape(-1, k)

        self.ax3d.plot_surface(
            X,
            Y,
            Z,
            color=MANIFOLD_COLOR,
            rstride=1,
            cstride=1,
            shade=False,
            linewidth=0.5,
            edgecolor=[0.3, 0.3, 0.3],
            alpha=1,
            zorder=0,
        )

    def draw_tangent_vectors(self, vfields, time=0):
        for n, (mfld, ax, scale) in enumerate(
            zip((self.M2d, self.M3d), (self.ax2d, self.ax3d), self.tvecs_scale)
        ):
            for point in mfld.points:
                vec_one = (
                    get_tangent_vector(point, vectors_field=vfields[0]) * scale
                )
                vec_two = (
                    get_tangent_vector(point, vectors_field=vfields[1]) * scale
                )

                vec = (1 - time) * vec_one + time * vec_two

                X = [point.embedded[0], point.embedded[0] + vec[0]]
                Y = [point.embedded[1], point.embedded[1] + vec[1]]
                Z = [point.embedded[2], point.embedded[2] + vec[2]]

                if n == 1:
                    if point[1] > 0.4 and point[0] > 0.7:
                        continue
                    coords = np.array(_curvy_plane_3d(*point.coordinates))
                    if coords[-1] <= 0.29 and point[1] < 0.5:
                        continue

                ax.plot(
                    X,
                    Y,
                    Z,
                    lw=5,
                    color=[0.2, 0.2, 0.2],
                    zorder=500,
                    solid_capstyle="round",
                )
                ax.plot(
                    X,
                    Y,
                    Z,
                    lw=4,
                    color=salmon,
                    zorder=500,
                    solid_capstyle="round",
                )

                if n == 1:
                    ax.plot(
                        [point.embedded[0], point.embedded[0] + 1e-4],
                        [point.embedded[1], point.embedded[1] + 1e-4],
                        [point.embedded[2], point.embedded[2] + 1e-4],
                        lw=8,
                        color=[0.4, 0.4, 0.4],
                        zorder=400,
                        solid_capstyle="round",
                    )

            # scatter point place
            if n == 0:
                points_coords = np.vstack([pt.embedded for pt in mfld.points])
                ax.scatter(
                    points_coords[:, 0],
                    points_coords[:, 1],
                    points_coords[:, 2],
                    color=[0.4, 0.4, 0.4],
                    zorder=200,
                    s=80,
                    lw=0,
                    alpha=1,
                )

    def make_video(self):
        time = np.linspace(0, 3.5, 70)
        camera = Camera(self.f)
        vfields = [
            (vifield_one, vifield_two),
            (vifield_two, vfield_three),
            (vfield_three, vifield_one),
            (vifield_one, vifield_one),
        ]
        for t in track(time, total=len(time), description="Generating frames"):
            self.draw_plane_2d()
            self.draw_plane_3d()

            t0 = int(np.floor(t))
            self.draw_tangent_vectors(vfields[t0], time=t - t0)
            camera.snap()
        print("Saving animation")
        camera.animate().save("animation.mp4")


viz = PyplotVisualizer()
viz.make_video()
plt.show()
