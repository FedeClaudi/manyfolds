import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Point:
    coordinates: tuple
    embedding_map: Callable

    def __getitem__(self, i):
        return self.coordinates[i]

    def __len__(self):
        return len(self.coordinats)

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

        if p >= self.left and p < self.right:
            return True
        else:
            return False

    def sample(self, n=10):
        return list(np.linspace(self.l, self.r - 0.001, n))


@dataclass
class Map:
    name: str
    f: Callable
    inverse: Callable

    def __call__(self, x):
        if isinstance(x, Point):
            return self.f(x.coordinates)

        else:
            raise NotImplementedError


@dataclass
class Chart:
    U: Interval
    x: Map


@dataclass
class Set:
    name: str


@dataclass
class Manifold:
    M: Set
    charts: list
