from loguru import logger
from functools import partial
from numpy import pi

from manifold.topology import Point, Manifold, Map, Chart, Interval
from manifold.maps import constant
from manifold.base_function import BaseFunction
from manifold.maps import identity
from manifold.manifolds.base import BaseManifold


class Manifold2D(BaseManifold):
    """
        Base class for 2D manifolds e.g. sphere
    """

    d = 2
    vis_n_points = [8, 40]

    def __init__(self, embedding, n_sample_points):
        super().__init__(embedding, n_sample_points=n_sample_points)

    def sample(self, n=None, fill=False, full=False):
        n = n or self.n_sample_points
        if not isinstance(n, list):
            n = [n] * self.d

        logger.debug(f"Sampling manifold points with n={n}")

        # sample from each interval the maifold is defined over.
        points = []
        if full:
            I0 = self.manifold.M[0].sample(n[0])
            skip_first = False
        else:
            I0 = self.manifold.M[0].sample(n[0])
            skip_first = True
        I1 = self.manifold.M[1].sample(n[1])

        for s, p0 in enumerate(I0):
            for q, p1 in enumerate(I1):
                if (q == 0 or s == 0) and skip_first:
                    continue
                point = Point((p0, p1), self.embedding)
                if point not in points:
                    points.append(point)

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
            point.projected = chart.x(point)

    def get_base_functions(self, points=None):
        """
            For each point in the sampled manifold
            define a function I -> x(u) in the image of 
            a chart containint the point
        """
        points = points or self.points
        for point in points:
            point.base_functions = [
                BaseFunction(
                    point=point,
                    f=partial(constant, point.coordinates[0]),
                    domain_interval=point.chart.U[0],
                    dim_idx=0,
                ),
                BaseFunction(
                    point=point,
                    f=partial(constant, point.coordinates[1]),
                    domain_interval=point.chart.U[1],
                    dim_idx=1,
                ),
            ]


class Plane(Manifold2D):
    name = "R2"
    manifold = Manifold(
        M=[Interval("M_1", 0, 1), Interval("M_2", 0, 1)],
        charts=[
            Chart(
                1,
                [Interval("U_1_1", 0, 0.6), Interval("U_1_2", 0, 0.6)],
                Map("x_1", identity, identity),
            ),
            Chart(
                2,
                [Interval("U_2_1", 0.4, 1), Interval("U_2_2", 0, 0.6)],
                Map("x_2", identity, identity),
            ),
            Chart(
                3,
                [Interval("U_3_1", 0, 0.6), Interval("U_3_2", 0.4, 1)],
                Map("x_3", identity, identity),
            ),
            Chart(
                4,
                [Interval("U_4_1", 0.4, 1), Interval("U_4_2", 0.4, 1)],
                Map("x_4", identity, identity),
            ),
        ],
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Sphere(Manifold2D):
    name = "S2"
    manifold = Manifold(
        M=[Interval("M_1", 0, pi), Interval("M_2", 0, 2 * pi)],
        charts=[
            Chart(
                1,
                [
                    Interval("U_1_1", 0, 0.7 * pi),
                    Interval("U_1_2", 0, 1.5 * pi),
                ],
                Map("x_1", identity, identity),
            ),
            Chart(
                2,
                [Interval("U_3_1", 0.3, pi), Interval("U_3_2", 0, 1.5 * pi)],
                Map("x_2", identity, identity),
            ),
            Chart(
                3,
                [Interval("U_2_1", 0.3, pi), Interval("U_2_2", 0.5, 2 * pi)],
                Map("x_3", identity, identity),
            ),
            Chart(
                4,
                [
                    Interval("U_4_1", 0, 0.7 * pi),
                    Interval("U_4_2", 0.5, 2 * pi),
                ],
                Map("x_4", identity, identity),
            ),
        ],
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Torus(Manifold2D):
    name = "R2"
    manifold = Manifold(
        M=[Interval("M_1", 0, 2 * pi), Interval("M_2", 0, 2 * pi)],
        charts=[
            Chart(
                1,
                [
                    Interval("U_1_1", 0, 1.5 * pi),
                    Interval("U_1_2", 0, 1.5 * pi),
                ],
                Map("x_1", identity, identity),
            ),
            Chart(
                2,
                [
                    Interval("U_3_1", 0.5, 2 * pi),
                    Interval("U_3_2", 0, 1.5 * pi),
                ],
                Map("x_2", identity, identity),
            ),
            Chart(
                3,
                [
                    Interval("U_2_1", 0, 1.5 * pi),
                    Interval("U_2_2", 0.5, 2 * pi),
                ],
                Map("x_3", identity, identity),
            ),
            Chart(
                4,
                [
                    Interval("U_4_1", 0.5, 2 * pi),
                    Interval("U_4_2", 0.5, 2 * pi),
                ],
                Map("x_4", identity, identity),
            ),
        ],
    )

    vis_n_points = [20, 40]

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)
