from dataclasses import dataclass
from typing import Callable
import numpy as np
from loguru import logger


from manifold.topology import Point, Interval


@dataclass
class BaseFunction:
    """
        Function for a point, used to take its derivative
        in embedding and get the base vectors for the tangent
        space at the point.
    """

    point: Point
    f: Callable

    embedded = None

    def embedd(self, x_range=0.1):
        """
            Embeds the function's domain in embedding space
            using the inverse of the point's chart's map and the 
            manifold embedding map
        """
        # get domain of the function
        l, r = (
            self.point.coordinates[0] - x_range,
            self.point.coordinates[0] + x_range,
        )
        domain = Interval(
            "domain",
            l if self.point.chart.U.contains(l) else self.point.chart.U.l,
            r if self.point.chart.U.contains(r) else self.point.chart.U.r,
        ).sample(n=100)
        logger.debug(
            f"Embedding base function for {self.point} with domain: {domain[0]:2f} -> {domain[-1]:.2f}"
        )

        self.embedded_point_index = 50  # index of the embedding point corresponding to the function's point

        # embedd
        embedded = [
            self.point.embedding_map(self.point.chart.x.inverse(x))
            for x in domain
        ]

        # reshape data
        self.embedded = np.vstack(
            [
                [e[0] for e in embedded],
                [e[1] for e in embedded],
                [e[2] for e in embedded],
            ]
        ).T

    @property
    def tangent_vector(self):
        """
            Returns the tangent vector in embedding space
        """
        if self.embedded is None:
            self.embedd()

        derivative = np.diff(self.embedded.T)

        return derivative[:, self.embedded_point_index].T
