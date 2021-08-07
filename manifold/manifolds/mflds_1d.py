from loguru import logger
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from manifold.topology import Point, Manifold, Map, Chart, Interval
from manifold.maps import (
    identity,
    subtract_pi,
    subtract_pi_inverse,
    smul_2,
    smul_2_inverse,
    smul_pi,
    smul_pi_inverse,
)
from manifold.base_function import BaseFunction
from manifold.manifolds.manifold import BaseManifold


class Manifold1D(BaseManifold):
    """
        Base class for 1D manifolds
    """

    d = 1
    vis_n_points = 300

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
                BaseFunction(point=point, f=self.base_functions_map,)
            ]

    def sample(self, n=None, fill=False, **kwargs):
        """
            Samples N points from the manifold's interval ensuring that they are not too close
        """
        # how many points
        n = n or self.n_sample_points
        if isinstance(n, list):
            n = n[0]
        logger.debug(f"Sampling manifold points with n={n+1}")

        # sample points
        points = [Point((k,), self.embedding) for k in self.M.sample(n)]

        # fill all data for points
        if fill:
            self._fill_points_data(points)
        return points

    def visualize_charts(self):
        """
            Takes point from the manifold domain and shows their projections in
            the charts domains
        """
        f, axes = plt.subplots(nrows=2, figsize=(16, 9), sharex=True)

        for n, point in enumerate(self.sample(n=50, fill=True)):
            axes[0].scatter(point.coordinates[0], 1, c=n, vmin=0, vmax=50)
            axes[1].scatter(
                point.chart_coordinates, point.chart.idx, c=n, vmin=0, vmax=50
            )
        axes[1].axhline(1, ls=":", color=[0.2, 0.2, 0.2], zorder=-1)
        axes[1].axhline(2, ls=":", color=[0.2, 0.2, 0.2], zorder=-1)

        axes[0].set(yticks=[], xlabel="M domain")
        axes[1].set(
            yticks=np.arange(len(self.manifold.charts)) + 1,
            yticklabels=[f"chart {c.idx}" for c in self.manifold.charts],
        )


class Circle(Manifold1D):
    name = "S^1"
    manifold = Manifold(
        M=Interval("M", 0, 2 * pi),
        charts=[
            Chart(1, Interval("U_1", 0, pi), Map("x_1", identity, identity),),
            Chart(
                2,
                Interval("U_2", pi, 2 * pi, left_open=True),
                Map("x_2", subtract_pi, subtract_pi_inverse),
            ),
        ],
    )
    _full = False
    base_functions_map = Map(
        "pi scalar multiplication", smul_pi, smul_pi_inverse
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Line(Manifold1D):
    name = "R^1"
    manifold = Manifold(
        M=Interval("M", 0, 1),
        charts=[
            Chart(
                1, Interval("U_1", 0, 1), Map("x_1", smul_2, smul_2_inverse),
            ),
        ],
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)
