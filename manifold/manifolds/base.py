import numpy as np
from loguru import logger

from myterial import blue_grey, grey_dark, grey

from manifold.topology import Point
from manifold.visualize import make_3D_ax


class BaseManifold:
    def __init__(self, embedding, n_sample_points=10):
        self.embedding = embedding

        if self.d == 1:
            # sample 1D manifold
            self.n_sample_points = n_sample_points + 2

            # sample points in the manifold
            self.points = self.sample()[1:-1]
        else:
            self.n_sample_points = n_sample_points
            self.points = self.sample()
        logger.debug(f"")

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

    def _fill_points_data(self, points):
        """
            Fills points data with chart projections, 
            base functions etc
        """
        # embedd:
        for point in points:
            point.embedded = self.embedding(point)

        # get base functions
        self.project_with_charts(points)
        self.get_base_functions(points)

    def get_chart_from_point(self, point):
        """
            Returns the first of the manifold's chart that 
            contains a given point in its image. Also stores
            the chart in the point
        """
        for chart in self.manifold.charts:
            if chart.contains(point):
                point.chart = chart
                return chart
        raise ValueError(f"No chart contains the point: {point}")

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
            for p in self.sample(n=self.vis_n_points, full=True)
        ]
        self.embedded = np.vstack(
            [p.coordinates for p in self.embedded_points_vis]
        )

    def visualize_embedded(self):
        ax = make_3D_ax()

        for p in self.points:
            ax.scatter(
                *p.embedded,
                s=100,
                # color=blue,
                c=p.chart.idx,
                vmin=-1,
                vmax=4,
                zorder=2,
                edgecolors=grey,
                lw=0.5,
            )

        if self.d == 1:
            ax.plot(
                self.embedded[0, 0],
                self.embedded[:, 1],
                self.embedded[:, 2],
                lw=1.5,
                color=grey,
                zorder=-1,
            )
        else:
            for n in range(self.vis_n_points[1]):
                idxs = [
                    k + self.vis_n_points[0] * n
                    for k in range(self.vis_n_points[0])
                    if k + self.vis_n_points[0] * n < self.embedded.shape[0]
                ]
                ax.plot(
                    self.embedded[idxs, 0],
                    self.embedded[idxs, 1],
                    self.embedded[idxs, 2],
                    lw=1.5,
                    color=grey,
                    zorder=-1,
                )
        return ax

    def visualize_base_functions_at_point(self, ax, x_range=0.2, scale=0.2):
        """
            For a given point in the manifold it projects the base functions
            in the image of the points chart to the embedding. This is done by taking each 
            point in the domain of the function, passing it through the inverse chart map
            and then the embedding
        """
        if not isinstance(x_range, list):
            x_range = [x_range] * self.d

        for point in self.points:
            for fn in point.base_functions:
                # plot the function
                fn.embedd(x_range=x_range[fn.dim_idx])
                ax.plot(
                    fn.embedded[:, 0],
                    fn.embedded[:, 1],
                    fn.embedded[:, 2],
                    lw=5,
                    color=grey_dark,
                )
                # ax.scatter(
                #     fn.embedded[0, 0],
                #     fn.embedded[0, 1],
                #     fn.embedded[0, 2],
                #     s=100,
                #     color='m',
                # )
                # ax.scatter(
                #     *fn.point.embedded,
                #     s=100,
                #     color='k',
                # )
                # ax.scatter(
                #     fn.embedded[-1, 0],
                #     fn.embedded[-1, 1],
                #     fn.embedded[-1, 2],
                #     s=100,
                #     color='r',
                # )

                # plot the scaled tangent vector at the point
                vector = fn.tangent_vector
                vector = vector / np.linalg.norm(vector) * scale
                ax.plot(
                    [fn.point.embedded[0], fn.point.embedded[0] + vector[0]],
                    [fn.point.embedded[1], fn.point.embedded[1] + vector[1]],
                    [fn.point.embedded[2], fn.point.embedded[2] + vector[2]],
                    lw=4,
                    color=blue_grey,
                )

    # ------------------------ to implement in subclasses ------------------------ #
    def project_with_charts(self, points=None):
        """
            For each point it projects the point to the image
            of a chart map for a chart containing the point
        """
        raise NotImplementedError("project_with_charts Method not implemented")

    def get_base_functions(self, points=None):
        """
            For each point in the sampled manifold
            define a function I -> x(u) in the image of 
            a chart containint the point
        """
        raise NotImplementedError("get_base_functions Method not implemented")

    def sample(self, n=None, fill=False):
        """
            Samples N points from the manifold's interval ensuring that they are not too close
        """
        raise NotImplementedError("sample Method not implemented")
