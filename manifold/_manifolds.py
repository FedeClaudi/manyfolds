import matplotlib.pyplot as plt
from loguru import logger
from functools import partial

from myterial import blue_grey

from manifold.topology import Point
from manifold.maps import constant
from manifold.base_function import BaseFunction

color = "#c3c3db"
grey = [0.6, 0.6, 0.6]


class Manifold1D:
    d = 1

    def __init__(self, embedding, n_sample_points=10):
        self.n_sample_points = n_sample_points + 1
        self.embedding = embedding

        # sample points in the manifold
        self.points = self.sample()[:-1]

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

    def project_with_charts(self):
        """
            For each point it projects the point to the image
            of a chart map for a chart containing the point
        """
        for point in self.points:
            chart = self.get_chart_from_point(point)
            point.projected = chart.x(point)

    def get_base_functions(self):
        """
            For each point in the sampled manifold
            define a function I -> x(u) in the image of 
            a chart containint the point
        """
        for point in self.points:
            point.base_functions = [
                BaseFunction(point, partial(constant, point.coordinates[0]))
            ]

    def sample(
        self, n=None,
    ):
        """
            Samples N points from the manifold's interval ensuring that they are not too close
        """
        # how many points
        n = n or self.n_sample_points
        logger.debug(f"Sampling manifold points with n={n}")

        # sample points
        return [Point((k,), self.embedding) for k in self.M.sample(n)]

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

    def visualize_embedded(self):
        plt.figure()
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
            [p.coordinates[0] for p in self.embedded_points_vis],
            [p.coordinates[1] for p in self.embedded_points_vis],
            [p.coordinates[2] for p in self.embedded_points_vis],
            lw=1.5,
            color=grey,
            zorder=-1,
        )

        return ax

    def visualize_base_functions_at_point(self, ax, x_range=0.2):
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
                vector = fn.tangent_vector * 50
                ax.plot(
                    [fn.point.embedded[0], fn.point.embedded[0] + vector[0]],
                    [fn.point.embedded[1], fn.point.embedded[1] + vector[1]],
                    [fn.point.embedded[2], fn.point.embedded[2] + vector[2]],
                    lw=2,
                    color=blue_grey,
                )
