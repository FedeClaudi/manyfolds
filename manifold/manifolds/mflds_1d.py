import matplotlib.pyplot as plt
from loguru import logger
from functools import partial
import numpy as np
from numpy import pi

from myterial import blue_grey

from manifold.topology import Point, Manifold, Map, Chart, Interval
from manifold.maps import constant
from manifold.base_function import BaseFunction
from manifold.maps import identity


color = "#c3c3db"
grey = [0.6, 0.6, 0.6]


class Manifold1D:
    """
        Base class for 1D manifolds
    """

    d = 1

    def __init__(self, embedding, n_sample_points=10):
        self.n_sample_points = n_sample_points + 2
        self.embedding = embedding

        # sample points in the manifold
        self.points = self.sample()[1:-1]

        # project with charts
        self.project_with_charts()

        # embedd
        self.embedd()

    @property
    def n(self):
        """
            Dimensionality of the embedding
        """
        return len(self.points[0].embedded)

    @property
    def M(self):
        """ short hand for self.manifold.M """
        return self.manifold.M

    def get_chart_from_point(self, point):
        """
            Returns the first of the manifold's chart that 
            contains a given point in its image. Also stores
            the chart in the point
        """
        for chart in self.manifold.charts:
            if chart.U.contains(point):
                point.chart = chart
                return chart
        raise ValueError(f"No chart contains the point: {point}")

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
                BaseFunction(point, partial(constant, point.coordinates[0]))
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
            # embedd:
            for point in points:
                point.embedded = self.embedding(point)

            # get base functions
            self.project_with_charts(points)
            self.get_base_functions(points)

        return points

    def embedd(self):
        """
            Embed N randomly sampled points of the manifold's set
            using the embedding function
        """
        # map each sampled point to the embedding
        for point in self.points:
            point.embedded = self.embedding(point)

        # get more points for visualization
        self.embedded_points_vis = [
            Point(self.embedding(p), self.embedding)
            for p in self.sample(n=100)
        ]

        self.embedded = np.vstack(
            [p.coordinates for p in self.embedded_points_vis]
        )

    def visualize_embedded(self):
        plt.figure(figsize=(9, 9))
        ax = plt.axes(projection="3d")

        for p in self.points:
            ax.scatter(
                *p.embedded,
                s=30,
                color=color,
                zorder=2,
                edgecolors=grey,
                lw=0.5,
            )

        ax.plot(
            self.embedded[:, 0],
            self.embedded[:, 1],
            self.embedded[:, 2],
            lw=1.5,
            color=grey,
            zorder=-1,
        )

        return ax

    def visualize_base_functions_at_point(self, ax, x_range=0.2, scale=100):
        """
            For a given point in the manifold it projects the base functions
            in the image of the points chart to the embedding. This is done by taking each 
            point in the domain of the function, passing it through the inverse chart map
            and then the embedding
        """
        for point in self.points:
            for fn in point.base_functions:
                # plot the function
                fn.embedd(x_range=x_range)
                ax.plot(
                    fn.embedded[:, 0],
                    fn.embedded[:, 1],
                    fn.embedded[:, 2],
                    lw=3,
                    color=[0.4, 0.4, 0.4],
                )

                # plot the tangent vector at the point
                vector = fn.tangent_vector * scale
                ax.plot(
                    [fn.point.embedded[0], fn.point.embedded[0] + vector[0]],
                    [fn.point.embedded[1], fn.point.embedded[1] + vector[1]],
                    [fn.point.embedded[2], fn.point.embedded[2] + vector[2]],
                    lw=5,
                    color=blue_grey,
                )


class Circle(Manifold1D):
    name = "S_1"
    manifold = Manifold(
        M=Interval("M", 0, 2 * pi),
        charts=[
            Chart(
                Interval("U_1", 0, 1.5 * pi), Map("x_1", identity, identity),
            ),
            Chart(
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
            Chart(Interval("U_1", 0, 0.7), Map("x_1", identity, identity),),
            Chart(Interval("U_2", 0.3, 1), Map("x_2", identity, identity),),
        ],
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)
