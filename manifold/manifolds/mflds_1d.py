from loguru import logger
from functools import partial
from numpy import pi

from manifold.topology import Point, Manifold, Map, Chart, Interval
from manifold.maps import constant
from manifold.base_function import BaseFunction
from manifold.maps import identity
from manifold.manifolds.base import BaseManifold


class Manifold1D(BaseManifold):
    """
        Base class for 1D manifolds
    """

    d = 1
    vis_n_points = 100

    def __init__(self, embedding, n_sample_points):
        super().__init__(embedding, n_sample_points=n_sample_points)

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
                    domain_interval=point.chart.U,
                    dim_idx=0,
                )
            ]

    def sample(self, n=None, fill=False):
        """
            Samples N points from the manifold's interval ensuring that they are not too close
        """
        # how many points
        n = n or self.n_sample_points
        logger.debug(f"Sampling manifold points with n={n}")

        # sample points
        points = [Point((k,), self.embedding) for k in self.M.sample(n)]

        # fill all data for points
        if fill:
            self._fill_points_data(points)
        return points


class Circle(Manifold1D):
    name = "S_1"
    manifold = Manifold(
        M=Interval("M", 0, 2 * pi),
        charts=[
            Chart(
                1,
                Interval("U_1", 0, 1.5 * pi),
                Map("x_1", identity, identity),
            ),
            Chart(
                2,
                Interval("U_2", 0.5 * pi, 2 * pi),
                Map("x_2", identity, identity),
            ),
        ],
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Line(Manifold1D):
    name = "R_1"
    manifold = Manifold(
        M=Interval("M", 0, 1),
        charts=[
            Chart(1, Interval("U_1", 0, 0.7), Map("x_1", identity, identity),),
            Chart(2, Interval("U_2", 0.3, 1), Map("x_2", identity, identity),),
        ],
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)
