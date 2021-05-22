from dataclasses import dataclass
import numpy as np
from loguru import logger


from manifold.topology import Point, Interval, Map


@dataclass
class BaseFunction:
    """
        Function for a point, used to take its derivative
        in embedding and get the base vectors for the tangent
        space at the point.
    """

    point: Point
    f: Map

    embedded = None

    def embedd(self, x_range=0.1):
        """
            Embeds the function's domain in embedding space
            using the inverse of the point's chart's map and the 
            manifold embedding map
        """
        N = 100  # number of samples over the domain
        self.embedded_point_index = int(
            N / 2
        )  # index of the embedding point corresponding to the function's point

        # get domain of the function by using the inverse map
        point_in_domain = self.f.inverse(self.point.chart_coordinates)

        _range = x_range / 2
        domain = np.array(
            Interval(
                "", point_in_domain - _range, point_in_domain + _range
            ).sample(n=N)
        )

        # get points in the embedding
        # 1. map the domain of f to the chart's local coordinates
        chart_coords = self.f(domain)

        # 2. use the chart inverse map to go the manifold
        manifold_coords = self.point.chart.x.inverse(chart_coords).reshape(
            -1, 1
        )

        # 3. use the embedding map to the the coordinates in the embedding space
        self.embedded = np.apply_along_axis(
            self.point.embedding_map, 1, manifold_coords
        )

    @property
    def tangent_vector(self):
        """
            Returns the tangent vector in embedding space
        """
        if self.embedded is None:
            self.embedd()

        derivative = np.diff(self.embedded.T)
        if np.linalg.norm(derivative) == 0:
            logger.warning(
                f"Tangent vector for base function {self} is vanishing"
            )

        return derivative[:, self.embedded_point_index].T


@dataclass
class BaseFunction2D:
    """
        Function for a point, used to take its derivative
        in embedding and get the base vectors for the tangent
        space at the point.
    """

    point: Point
    f: Map
    dim_idx: int

    embedded = None

    def embedd(self, x_range=0.1):
        """
            Embeds the function's domain in embedding space
            using the inverse of the point's chart's map and the 
            manifold embedding map
        """
        N = 100  # number of samples over the domain
        self.embedded_point_index = int(
            N / 2
        )  # index of the embedding point corresponding to the function's point

        # get domain of the function by using the inverse map
        point_in_domain = self.f.inverse(self.point.chart_coordinates)[
            self.dim_idx
        ]

        _range = x_range / 2
        domain = np.array(
            Interval(
                "", point_in_domain - _range, point_in_domain + _range
            ).sample(n=N)
        )

        # get points in the embedding
        # 1. map the domain of f to the chart's local coordinates
        # keep one dymension and vary the other
        fixed = self.point.chart_coordinates[1 - self.dim_idx]
        chart_coords = np.ones((N, self.point.d)) * fixed
        chart_coords[:, self.dim_idx] = self.f(domain)[:N]

        # 2. use the chart inverse map to go the manifold
        manifold_coords = self.point.chart.x.inverse(chart_coords)

        # 3. use the embedding map to the the coordinates in the embedding space
        self.embedded = np.apply_along_axis(
            self.point.embedding_map, 1, manifold_coords
        )

    @property
    def tangent_vector(self):
        """
            Returns the tangent vector in embedding space
        """
        if self.embedded is None:
            self.embedd()

        derivative = np.diff(self.embedded.T)
        if np.linalg.norm(derivative) == 0:
            logger.warning(
                f"Tangent vector for base function {self} is vanishing"
            )

        return derivative[:, self.embedded_point_index].T
