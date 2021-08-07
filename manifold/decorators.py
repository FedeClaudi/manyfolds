import numpy as np
from manifold.topology import Point


def return_many(func):
    def inner(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0]
        else:
            return out

    return inner


"""
    Wrappers for embedding maps (\\phi) python functions,
    to facilitate handling different types of inputs
"""


def parse(func):
    """
        Wrapper for 1D manifolds embedding to work
        with either a single float or a Point
        of 1 coordinate
    """

    def inner(*args):
        if len(args) == 1:
            p = args[0]

            if not isinstance(p, float):
                p = p[0]
            return func(p)
        else:
            return func(*args)

    return inner


def parse2D(func):
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
