from numpy import sin, cos
import numpy as np


from manifold.topology import Point

# --------------------------------- wrappers --------------------------------- #


def to_coordinates(func):
    """
        Wrapper for 1D manifolds embedding to work
        with either a single float or a Point
        of 1 coordinate
    """

    def inner(p):
        if not isinstance(p, float):
            p = p[0]
        return func(p)

    return inner


def to_coordinates2D(func):
    """
        Wrapper for 2D manifolds embedding to work
        with either a Point as argument or a list of floats
    """

    def inner(p, *args):
        if isinstance(p, np.ndarray):
            return func(*p)
        elif not isinstance(p, Point):
            return func(p, *args)
        else:
            return func(*p.coordinates)

    return inner


# ----------------------------------- line ----------------------------------- #
@to_coordinates
def line_to_r3_flat(p):
    return (p, p, p)


@to_coordinates
def line_to_r3(p):
    return (sin(2 * p) - 0.5, sin(p) * 2 - 1, -cos(p) * 4 + 3)


# ---------------------------------- circle ---------------------------------- #
@to_coordinates
def circle_to_r3_flat(p):
    """
        Embedds a circle in 3D but keeping the circle flat in one dimension
    """
    return (sin(p), cos(p), 1)


@to_coordinates
def circle_to_r3(p):
    return (sin(p), cos(p) / 2, sin(p))


# ---------------------------------- sphere ---------------------------------- #
@to_coordinates2D
def sphere_to_r3(p0, p1):
    return (sin(p0) * cos(p1), sin(p0) * sin(p1), cos(p0))


# ----------------------------------- plane ---------------------------------- #
@to_coordinates2D
def plane_to_r3_flat(p0, p1):
    return (p0 - 0.5, p1 - 0.5, p0 + p1 - 1)


@to_coordinates2D
def plane_to_r3(p0, p1):
    return (2 * p0 - 1, 2 * sin(p1) - 1, 2 * cos(2 * (p0 + p1 - 1)) - 1)


# ----------------------------------- torus ---------------------------------- #
@to_coordinates2D
def torus_to_r3(p0, p1):
    R = 0.75  # torus center -> tube center
    r = 0.5  # tubre radius
    return (
        (R + r * cos(p0)) * cos(p1),
        (R + r * cos(p0)) * sin(p1),
        r * sin(p0),
    )
