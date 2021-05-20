from numpy import pi

from manifold.topology import Manifold, Map, Chart, Interval, Interval2D
from manifold._manifolds import Base
from manifold.maps import identity


class Circle(Base):
    name = 'S_1'
    manifold = Manifold(
        M = Interval('M', 0, 2*pi),
        charts = [
            Chart(
                Interval('U_1', 0, 1.5 * pi),
                Map('x_1', identity),
            ),
            Chart(
                Interval('U_2', 0.5 * pi, 2 * pi),
                Map('x_2', identity),
            )
        ]
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)


class Sphere(Base):
    name = 'S2'

    manifold = Manifold(
        M = Interval2D(
            Interval('M_1', 0, pi),
            Interval('M_2', 0, 2*pi),
        ),
        charts = []
    )

    def __init__(self, embedding, n_sample_points=10):
        super().__init__(embedding, n_sample_points=n_sample_points)
