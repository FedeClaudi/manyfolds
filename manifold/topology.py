import numpy as np
from dataclasses import dataclass
from typing import Callable, Union


@dataclass
class Point:
    coordinates: tuple
    embedding_map: Callable

    def __getitem__(self, i):
        return self.coordinates[i]

    def __len__(self):
        return len(self.coordinates)

    def __eq__(self, other):
        delta = np.linalg.norm(
            np.array(self.coordinates) - np.array(other.coordinates)
        )
        return delta < 0.1

    @property
    def d(self):
        return len(self)

    @property
    def acord(self):
        """ coordinates as arrays """
        return np.array(self.coordinates)


@dataclass
class Interval:
    name: str
    left: float
    right: float
    left_open: bool = False  # if its open at left or right

    @property
    def l(self):
        return self.left

    @property
    def r(self):
        return self.right

    def contains(self, point):
        """
            Check if a point is contained in the interval
        """
        if not isinstance(point, float):
            p = point.coordinates[0]
        else:
            p = point

        if not self.left_open:
            contained = p >= self.left and p < self.right
        else:
            contained = p > self.left and p <= self.right

        return contained

    def sample(self, n=10, l_offset=0, r_offset=0):
        return list(
            np.linspace(self.l + l_offset, self.r - r_offset - 0.001, n + 1)
        )


@dataclass
class Map:
    name: str
    f: Callable
    inverse: Callable

    def __call__(self, x):
        if isinstance(x, Point):
            return self.f(x.coordinates)
        elif isinstance(x, np.ndarray):
            return self.f(x)
        elif isinstance(x, (tuple, list)):
            return [self.f(xx) for xx in x]
        else:
            raise NotImplementedError


@dataclass
class Chart:
    idx: int
    U: Union[Interval, list]
    x: Map

    def contains(self, point):
        if isinstance(self.U, Interval):
            # 1D manifold
            return self.U.contains(point)
        else:
            contained = [
                interval.contains(point.coordinates[n])
                for n, interval in enumerate(self.U)
            ]
            return np.all(contained)


@dataclass
class Manifold:
    M: Union[Interval, list]
    charts: list

    def contains(self, p):
        if isinstance(self.M, Interval):
            return self.M.contains(p)
        else:
            return np.all(interval.contains(p) for interval in self.M)
