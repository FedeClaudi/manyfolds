import numpy as np
from dataclasses import dataclass
from sympy import Function

@dataclass
class Point:
    coordinates: tuple

    def __getitem__(self, i):
        return self.coordinates[i]

    def __len__(self):
        return len(self.coordinats)

    @property
    def d(self):
        return len(self)

    @property
    def acord(self):
        ''' coordinates as arrays '''
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


@dataclass
class Interval2D:
    int1: Interval
    int2: Interval

@dataclass
class Map:
    name: str
    f: Function

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
