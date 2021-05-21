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
    domain_interval: Interval
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

        # get domain of the function
        l, r = (
            self.point.coordinates[0] - x_range,
            self.point.coordinates[0] + x_range,
        )
        domain = Interval(
            "domain",
            l if self.domain_interval.contains(l) else self.domain_interval.l,
            r if self.domain_interval.contains(r) else self.domain_interval.r,
        ).sample(n=N)
        logger.debug(
            f"Embedding base function for point @ {[round(x,2) for x in self.point.coordinates]} with chart: {self.point.chart.idx} with domain: {domain[0]:2f} -> {domain[-1]:.2f}"
        )

        # embedd
        if self.point.d == 1:
            _fixed = np.ones(self.point.d)
        else:
            _fixed = (
                np.ones(self.point.d)
                * self.point.coordinates[1 - self.dim_idx]
            )

        embedded = []
        for p in domain:
            # one dimension is fixed and one varies
            _point = _fixed.copy()
            _point[self.dim_idx] = self.point.chart.x.inverse(p)

            embedded.append(self.point.embedding_map(_point))

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
        if np.linalg.norm(derivative) == 0:
            logger.warning(
                f"Tangent vector for base function {self} is vanishing"
            )

        return derivative[:, self.embedded_point_index].T
