from numpy import sin, cos, pi
import numpy as np
import numpy.random as npr
from scipy.stats import ortho_group
from functools import partial

from manifold.topology import Point

# --------------------------------- wrappers --------------------------------- #


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


# ---------------------------------------------------------------------------- #
#                                    R^N = 3                                   #
# ---------------------------------------------------------------------------- #

# ----------------------------------- line ----------------------------------- #
@parse
def line_to_r3_flat(p):
    return (p, p, p)


@parse
def line_to_r3(p):
    return (sin(2 * p) - 0.5, sin(p) * 2 - 1, -cos(p) * 4 + 3)


@parse
def helix(p):
    return (cos(2 * pi * p), sin(2 * pi * p), 2 * p - 1)


# ---------------------------------- circle ---------------------------------- #
@parse
def circle_to_r3_flat(p):
    """
        Embedds a circle in 3D but keeping the circle flat in one dimension
    """
    return (sin(p), cos(p), 1)


@parse
def circle_to_r3(p):
    return (sin(p), cos(p) / 2, sin(p))


# ---------------------------------- sphere ---------------------------------- #
@parse2D
def sphere_to_r3(p0, p1):
    return (sin(p0) * cos(p1), sin(p0) * sin(p1), cos(p0))


# ----------------------------------- plane ---------------------------------- #
@parse2D
def plane_to_r3_flat(p0, p1):
    return (p0 - 0.5, p1 - 0.5, p0 + p1 - 1)


@parse2D
def plane_to_r3(p0, p1):
    return (2 * p0 - 1, 2 * sin(p1) - 1, 2 * cos(2 * (p0 + p1 - 1)) - 1)


# ----------------------------------- torus ---------------------------------- #
@parse2D
def torus_to_r3(p0, p1):
    R = 0.75  # torus center -> tube center
    r = 0.5  # tubre radius
    return (
        (R + r * cos(p0)) * cos(p1),
        (R + r * cos(p0)) * sin(p1),
        r * sin(p0),
    )


# ---------------------------------------------------------------------------- #
#                                    R^N > 3                                   #
# ---------------------------------------------------------------------------- #

# ----------------------------------- line ----------------------------------- #
@parse
def line_to_rn(funcs, factors, scales, p):
    """
        Embeds points of a line manifold in high D space
        with a set of trigonometric functions
    """
    coords = np.hstack(
        [f(s * p) * scale for s, f, scale in zip(factors, funcs, scales)]
    )
    return tuple(coords)


def prepare_line_to_rn(n=64):
    funcs = npr.choice((sin, cos), size=n)
    factors = npr.choice((1, 2), size=n)
    scales = npr.uniform(0.2, 0.6, size=n)
    return partial(line_to_rn, funcs, factors, scales)


@parse
def line_to_rn_flat(p, n=64):
    """
        Embeds points of a line manifold in high D space
        with a set of trigonometric functions
    """
    coords = []
    scale = np.linspace(0.8, 0.2, n + 1)
    for dim in range(n):
        coords.append(p * scale[n])
    return tuple(coords)


# ---------------------------------- circle ---------------------------------- #
@parse
def circle_to_rn(p, n=64):
    coords = []
    func = npr.choice((sin, cos), size=n + 1)
    for dim in range(n):
        coords.append(func[n](p))
    return tuple(coords)


@parse
def circle_to_rn_flat(v, m, p):
    """
        Embedding of a circle in a random 2D plane in R^n
        from: https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    """
    # define points on the circle of radius 1: 𝑟(cos𝑡)𝐯1+𝑟(sin𝑡)𝐯
    coords = cos(p) * v + sin(p) * m

    return tuple(coords)


def prepare_circle_embedding(n=64):
    x = ortho_group.rvs(n)
    v = x[:, 0]
    m = x[:, 1]

    return partial(circle_to_rn_flat, v, m)


# ----------------------------------- torus ---------------------------------- #
def torus_to_rn(mtx, p):
    """
        Embedd a torus by first embedding it in
        R3 and then using a linear transformatin
        to Rn
    """
    torus_3d = torus_to_r3(p)
    embedded = mtx @ np.array(torus_3d)
    return tuple(embedded)


def prepare_torus_to_rn(n=64):
    mtx = np.random.rand(n, 3)

    return partial(torus_to_rn, mtx)
