import numpy as np
from rich.table import Table

# from rich import print
from loguru import logger

from myterial import pink

from manifold.topology import Point, Map
from manifold.maps import identity
from manifold.manifolds import vectors_fields


class BaseManifold:
    actors = []  # for visualization
    # maps used by base functions
    base_functions_map = Map("id", identity, identity)

    # maps to define vector fields on the manifold
    vectors_field = vectors_fields.identity

    def __init__(self, embedding, n_sample_points=10):
        self.embedding = embedding

        if self.d == 1:
            # sample 1D manifold
            self.n_sample_points = n_sample_points + 2

            # sample points in the manifold
            self.points = self.sample()[:-1]
        else:
            self.n_sample_points = n_sample_points
            self.points = self.sample()

        # project with charts
        self.project_with_charts()

        # embedd
        self.embedd()

        # get base functions at each point
        self.get_base_functions()

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

    @property
    def CoM(self):
        return np.mean(self.embedded, axis=0)

    def print_embedding_bounds(self):
        """
            it prints the boundary values of the embedded manifold
            along each dimensions
        """
        tb = Table(box=None, header_style=f"bold {pink}")
        tb.add_column("Dimension", style="dim", justify="right")
        tb.add_column("Bounds", justify="left")

        lows, highs = [], []
        for dim in range(self.embedded.shape[1]):
            low = self.embedded[:, dim].min()
            high = self.embedded[:, dim].max()
            lows.append(low)
            highs.append(high)
            tb.add_row(str(dim), f"min: {low:.2f} | max: {high:.2f}")
        # print(tb)
        logger.debug(
            f"Manifold bounds: low: {np.min(lows):.2f} | max: {np.max(highs):.2f}"
            + f"   Average bounds: {np.mean(lows):.2f} | {np.mean(highs):.2f}"
        )

        # get center of mass
        logger.debug(f"Manifold CoM: {self.CoM.mean():.2f}")

    def _fill_points_data(self, points):
        """
            Fills points data with chart projections, 
            base functions etc
        """
        # embedd:
        for point in points:
            point.embedded = np.array(self.embedding(point))

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
            point.embedded = np.array(self.embedding(point))

        # get more points for visualization
        self.embedded_points_vis = [
            Point(self.embedding(p), self.embedding)
            for p in self.sample(n=self.vis_n_points, full=True)
        ]
        self.embedded = np.vstack(
            [p.coordinates for p in self.embedded_points_vis]
        )

        # make sure that the manifold is centered at the origin
        if np.any(self.CoM != 0):
            self.embedded = self.embedded - self.CoM

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

    def visualize_charts(self):
        raise NotImplementedError('"visualize_charts" not implemented ')
