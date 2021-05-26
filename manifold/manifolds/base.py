import numpy as np
from rich.table import Table
from rich import print
from loguru import logger
from vedo import Sphere, Line, show, Tube, Torus, recoSurface

from myterial import grey_dark, grey, blue, green, pink

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
        print(tb)
        logger.debug(
            f"Manifold bounds: low: {np.min(lows):.2f} | max: {np.max(highs):.2f}"
        )

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

        for p in self.points:
            self.actors.append(Sphere(p.embedded, r=0.05, c=blue, res=12,))

        if self.d == 1:
            self.actors.append(Line(self.embedded, lw=4, c=grey,))
        else:
            if self.name == "S2":
                # plot a sphere
                self.actors.append(Sphere(r=0.75, c=grey).wireframe())

            elif self.name == "Cy":
                # plot a cylinder
                raise NotImplementedError

            elif self.name == "T2":
                # plot a torus
                self.actors.append(
                    Torus(r=0.5, thickness=0.25, c="grey", res=20,)
                    .wireframe()
                    .lw(1)
                )

            else:
                # plot points
                self.actors.append(
                    recoSurface(self.embedded, dims=(20, 20, 20), radius=0.5)
                    .c(grey)
                    .wireframe()
                    .lw(1)
                    .clean()
                )

    def visualize_base_functions_at_point(self, x_range=0.2, scale=0.2):
        """
            For a given point in the manifold it projects the base functions
            in the image of the points chart to the embedding. This is done by taking each 
            point in the domain of the function, passing it through the inverse chart map
            and then the embedding
        """
        if not isinstance(x_range, list):
            x_range = [x_range] * self.d

        for point in self.points:
            weights = self.vectors_field(point)
            vectors = []
            for n, fn in enumerate(point.base_functions):
                # plot the function
                fn.embedd(x_range=x_range[fn.dim_idx])
                self.actors.append(Tube(fn.embedded, r=0.02, c=grey_dark,))

                # plot the scaled tangent vector at the point
                vectors.append(fn.tangent_vector * scale * weights[n])

            vector = np.sum(np.vstack(vectors), 0)
            pts = np.vstack(
                [
                    [fn.point.embedded[0], fn.point.embedded[0] + vector[0]],
                    [fn.point.embedded[1], fn.point.embedded[1] + vector[1]],
                    [fn.point.embedded[2], fn.point.embedded[2] + vector[2]],
                ]
            ).T

            self.actors.append(Tube(pts, r=0.03, c=green,))

    def show(self):
        for actor in self.actors:
            actor.lighting("plastic")

        camera = dict(
            pos=[-0.025, -5.734, 4.018],
            focalPoint=[0.115, -0.647, 0.251],
            viewup=[-0.039, 0.595, 0.802],
            distance=6.331,
            clippingRange=[0.032, 32.461],
        )

        show(*self.actors, size="full", title=self.name, axes=1, camera=camera)

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
