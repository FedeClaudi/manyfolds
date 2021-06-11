from dataclasses import dataclass
import numpy as np

from manifold.topology import Point, Interval, Map
from manifold.tangent_vector import get_basis_tangent_vector


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
    dim_idx = 0
    N = 100  # number of samples over the domain

    @property
    def embedded_point_index(self):
        return int(self.N / 2) - 1

    def get_manifold_coordinates(self, x_range=0.1):
        # get domain of the function by using the inverse map
        point_in_domain = self.f.inverse(self.point.chart_coordinates)
        _range = x_range / 2
        domain = np.array(
            Interval(
                "", point_in_domain - _range, point_in_domain + _range
            ).sample(n=self.N)
        )

        # get points in the embedding
        # 1. map the domain of f to the chart's local coordinates
        chart_coords = self.f(domain)

        # 2. use the chart inverse map to go the manifold
        manifold_coords = self.point.chart.x.inverse(chart_coords).reshape(
            -1, 1
        )

        return manifold_coords

    def embedd(self, x_range=0.1):
        """
            Embeds the function's domain in embedding space
            using the inverse of the point's chart's map and the 
            manifold embedding map
        """
        manifold_coords = self.get_manifold_coordinates()

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

        return get_basis_tangent_vector(self.point, self)


class BaseFunction2D:
    """
        Function for a point, used to take its derivative
        in embedding and get the base vectors for the tangent
        space at the point.
    """

    embedded = None

    def __init__(self, point, f, dim_idx):
        self.point = point
        self.f = f
        self.dim_idx = dim_idx

        self.get_manifold_coordinates()

    def get_manifold_coordinates(self, x_range=0.2):
        N = 100  # number of samples over the domain
        self.embedded_point_index = int(
            N / 2
        )  # index of the embedding point corresponding to the function's point

        # get domain of the function by using the inverse map and get wehere the point is in the domain
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
        # keep one dimension fixed and vary the other
        fixed = self.point.chart_coordinates[1 - self.dim_idx]
        chart_coords = np.ones((N, self.point.d)) * fixed
        chart_coords[:, self.dim_idx] = self.f(domain)[:N]

        # 2. use the chart inverse map to go the manifold
        self.manifold_coords = self.point.chart.x.inverse(chart_coords)

    def embedd(self, x_range=0.1):
        """
            Embeds the function's domain in embedding space
            using the inverse of the point's chart's map and the 
            manifold embedding map
        """
        # 3. use the embedding map to the the coordinates in the embedding space
        self.embedded = (
            np.apply_along_axis(
                self.point.embedding_map, 1, self.manifold_coords
            )
            + self.point.shift
        )

    @property
    def tangent_vector(self):
        """
            Returns the tangent vector in embedding space
        """
        if self.embedded is None:
            self.embedd()

        return get_basis_tangent_vector(self.point, self,)
