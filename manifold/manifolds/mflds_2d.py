from loguru import logger
from numpy import pi
import matplotlib.pyplot as plt
import numpy as np

from manifold.topology import Point, Manifold, Map, Chart, Interval
from manifold.base_function import BaseFunction2D
from manifold import maps
from manifold.maps import identity
from manifold.manifolds.base import BaseManifold
from manifold.maths import cartesian_product


class Manifold2D(BaseManifold):
    """
        Base class for 2D manifolds e.g. sphere
    """

    _full = True
    d = 2
    vis_n_points = [8, 40]

    base_functions_map = [
        Map("id", maps.identity, maps.identity),
        Map("id", maps.identity, maps.identity),
    ]

    def __init__(self, embedding, n_sample_points):
        super().__init__(embedding, n_sample_points=n_sample_points)

    def sample(self, n=None, fill=False, full=None):
        full = full or self._full
        n = n or self.n_sample_points
        if not isinstance(n, list):
            n = [n] * self.d

        logger.debug(f"Sampling manifold points with n={[x+1 for x in n]}")

        # sample from each interval the maifold is defined over.
        points = []
        if not full:
            I0 = np.array(self.manifold.M[0].sample(n[0] + 2)[1:-1])
        else:
            I0 = np.array(self.manifold.M[0].sample(n[0]))
        I1 = np.array(self.manifold.M[1].sample(n[1]))

        points_coords = cartesian_product(I0, I1)
        points = [
            Point(tuple(points_coords[n, :]), self.embedding)
            for n in range(points_coords.shape[0])
        ]

        if fill:
            self._fill_points_data(points)

        return points

    def project_with_charts(self, points=None):
        """
            For each point it projects the point to the image
            of a chart map for a chart containing the point
        """
        points = points or self.points

        for point in points:
            chart = self.get_chart_from_point(point)
            point.chart_coordinates = chart.x(point)
            point.chart = chart

    def get_base_functions(self, points=None):
        """
            For each point in the sampled manifold
            define a function I -> x(u) in the image of 
            a chart containint the point
        """
        points = points or self.points
        for point in points:
            point.base_functions = [
                BaseFunction2D(
                    point=point, f=self.base_functions_map[0], dim_idx=0,
                ),
                BaseFunction2D(
                    point=point, f=self.base_functions_map[1], dim_idx=1,
                ),
            ]

    def visualize_charts(self):
        """
            Takes point from the manifold domain and shows their projections in
            the charts domains
        """
        f, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(16, 9), sharex=True, sharey=True
        )
        axes[0, 1].axis("off")
        axes = axes.flatten()

        for n, point in enumerate(
            self.sample(n=[10, 10], fill=True, full=True)
        ):
            axes[0].scatter(*point.coordinates, c=n, vmin=0, vmax=100)

            axes[point.chart.idx + 1].scatter(
                *point.chart_coordinates, c=n, vmin=0, vmax=100
            )

        axes[0].set(title="Manifold points")
        axes[2].set(title="Chart 1")
        axes[3].set(title="Chart 2")
        axes[4].set(title="Chart 3")
        axes[5].set(title="Chart 4")

        plt.show()


# ---------------------------------------------------------------------------- #
#                                   MANIFOLDS                                  #
# ---------------------------------------------------------------------------- #


class Plane(Manifold2D):
    name = "R^2"
    manifold = Manifold(
        M=[Interval("M_1", 0, 1), Interval("M_2", 0, 1)],
        charts=[
            Chart(
                1,
                [Interval("U_1_1", 0, 1), Interval("U_1_2", 0, 1)],
                Map("x_1", identity, identity),
            ),
        ],
    )
    _center_embedding = False
    vis_n_points = [30, 30]
    _full = True

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Sphere(Manifold2D):
    name = "S^2"
    manifold = Manifold(
        M=[Interval("M_1", 0, pi), Interval("M_2", 0, 2 * pi)],
        charts=[
            Chart(
                1,
                [Interval("U_1_1", 0, pi), Interval("U_1_2", 0, pi)],
                Map("x_1", maps.smul_pi_inverse, maps.smul_pi),
            ),
            Chart(
                2,
                [Interval("U_2_1", 0, pi), Interval("U_2_2", pi, 2 * pi)],
                Map("x_2", maps.sphere_U_2, maps.sphere_U_2_inverse),
            ),
        ],
    )

    _full = False
    vis_n_points = [5, 20]

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Cylinder(Manifold2D):
    name = "Cy"
    manifold = Manifold(
        M=[Interval("M_1", 0, 2 * pi), Interval("M_2", 0, 1)],
        charts=[
            Chart(
                1,
                [Interval("U_1_1", 0, pi), Interval("U_1_2", 0, 1)],
                Map("x_1", maps.cylinder_U_1, maps.cylinder_U_1_inverse),
            ),
            Chart(
                2,
                [Interval("U_2_1", pi, 2 * pi), Interval("U_2_2", 0, 1)],
                Map("x_2", maps.cylinder_U_2, maps.cylinder_U_2_inverse),
            ),
        ],
    )

    vis_n_points = [20, 5]
    _center_embedding = True
    _full = True

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Torus(Manifold2D):
    name = "T^2"
    manifold = Manifold(
        M=[Interval("M_1", 0, 2 * pi), Interval("M_2", 0, 2 * pi)],
        charts=[
            Chart(
                1,
                [Interval("U_1_1", 0, pi), Interval("U_1_2", 0, 2 * pi)],
                Map("x_1", maps.torus_U_1, maps.torus_U_1_inverse),
            ),
            Chart(
                2,
                [Interval("U_2_1", pi, 2 * pi), Interval("U_2_2", 0, 2 * pi)],
                Map("x_1", maps.torus_U_2, maps.torus_U_2_inverse),
            ),
        ],
    )

    # _full = False
    vis_n_points = [10, 50]

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)
