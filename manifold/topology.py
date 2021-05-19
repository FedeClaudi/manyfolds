from dataclasses import dataclass

@dataclass
class Point:
    coordinates: tuple

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

class Map:
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression

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
