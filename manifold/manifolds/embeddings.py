from numpy import sin, cos, pi
import numpy as np
from scipy.stats import ortho_group
from functools import partial

from manifold.topology import Point
from manifold.maths import ortho_normal_matrix

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
def helix_to_r3(p):
    return (cos(4 * pi * p) / 2, sin(4 * pi * p) / 2, p / 2 + 0.25)


# ---------------------------------- circle ---------------------------------- #
@parse
def circle_to_r3_flat(p):
    """
        Embedds a circle in 3D but keeping the circle flat in one dimension
    """
    return (sin(p), cos(p), 1)


@parse
def circle_to_r3_angled(p):
    return (sin(p), cos(p), sin(p))


@parse
def circle_to_r3(p):
    return (2 * sin(p), cos(p), cos(p) ** 2 + 0.5)


# ---------------------------------- sphere ---------------------------------- #
@parse2D
def sphere_to_r3(p0, p1):
    return (sin(p0) * cos(p1) * 0.75, sin(p0) * sin(p1) * 0.75, cos(p0) * 0.75)


# ----------------------------------- plane ---------------------------------- #
@parse2D
def plane_to_r3_flat(p0, p1):
    return (p0, p1, 0.5 * (p0 + p1))


@parse2D
def plane_to_r3(p0, p1):
    return (p0, sin(p1), (p1 - p0) ** 2)


# ----------------------------------- torus ---------------------------------- #
@parse2D
def torus_to_r3(p0, p1):
    R = 0.5  # torus center -> tube center
    r = 0.25  # tube radius
    return (
        (R + r * cos(p0)) * cos(p1),
        (R + r * cos(p0)) * sin(p1),
        r * sin(p0),
    )


# --------------------------------- cylinder --------------------------------- #
@parse2D
def cylinder_to_r3(p0, p1):
    return (sin(p0) / 2, cos(p0) / 2, p1 + 0.1)


@parse2D
def cylinder_to_r3_as_cone(p0, p1):
    k = p1 / 2 + 0.4
    return (k * sin(p0) / 2, k * cos(p0) / 2, p1 + 0.5)


# ---------------------------------------------------------------------------- #
#                                    R^N > 3                                   #
# ---------------------------------------------------------------------------- #


class TwoStepsEmbedding:
    def __init__(self, phi_1, n):
        # initialize projection matrix
        self.mtx = ortho_group.rvs(n)[:, :3]
        self.phi_1 = phi_1

    def __call__(self, p):
        embedded = self.mtx @ np.array(self.phi_1(p))
        return embedded


# ----------------------------------- line ----------------------------------- #
def prepare_line_to_rn(n=64):
    return TwoStepsEmbedding(line_to_r3, n)


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
def circle_to_rn(mtx, p):
    """
        Embedd a plane by first embedding it in
        R3 and then using a linear transformatin
        to Rn
    """
    circle_3d = circle_to_r3(p)
    embedded = mtx @ np.array(circle_3d)
    return tuple(embedded)


def prepare_circle_to_rn(n=64):
    mtx = ortho_normal_matrix(n, 3)
    return partial(circle_to_rn, mtx)


@parse
def circle_to_rn_flat(v, m, p):
    """
        Embedding of a circle in a random 2D plane in R^n
        from: https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    """
    # define points on the circle of radius 1: 𝑟(cos𝑡)𝐯1+𝑟(sin𝑡)𝐯
    coords = cos(p) * v + sin(p) * m

    return tuple(coords)


def prepare_circle_falt_to_rn(n=64):
    x = ortho_group.rvs(n)
    v = x[:, 0]
    m = x[:, 1]

    return partial(circle_to_rn_flat, v, m)


# ----------------------------------- helix ---------------------------------- #


@parse
def helix_to_rn(v, m, n, p):
    """
        Embeds points of a helix manifold in high D space
        with a set of trigonometric functions
    """
    if isinstance(p, Point):
        p = p.coordinates[0]
    embedded = cos(4 * pi * p) * v + sin(4 * pi * p) * m + p * n
    return tuple(embedded)


def prepare_helix_to_rn(n=64):
    x = ortho_group.rvs(n)
    v = x[:, 0]
    m = x[:, 1]
    n = x[:, 2]

    return partial(helix_to_rn, v, m, n)


# ---------------------------------- sphere ---------------------------------- #
def sphere_to_rn(mtx, p):
    """
        Embedd a sphere by first embedding it in
        R3 and then using a linear transformatin
        to Rn
    """
    sphere_3d = sphere_to_r3(p)
    embedded = mtx @ np.array(sphere_3d)
    return tuple(embedded)


def prepare_sphere_to_rn(n=64):
    mtx = np.random.rand(n, 3)

    return partial(sphere_to_rn, mtx)


# ---------------------------------- plane ---------------------------------- #
def plane_to_rn(mtx, p):
    """
        Embedd a plane by first embedding it in
        R3 and then using a linear transformatin
        to Rn
    """
    plane_3d = plane_to_r3(p)
    embedded = mtx @ np.array(plane_3d)
    return tuple(embedded)


def prepare_plane_to_rn(n=64):
    # mtx = ortho_group.rvs(n)[:, :3]
    mtx = ortho_normal_matrix(n, 3)
    return partial(plane_to_rn, mtx)


def flat_plane_to_rn(mtx, p):
    """
        Embedd a plane by first embedding it in
        R3 and then using a linear transformatin
        to Rn
    """
    plane_3d = plane_to_r3_flat(p)
    embedded = mtx @ np.array(plane_3d)
    return tuple(embedded)


def prepare_flat_plane_to_rn(n=64):
    mtx = ortho_group.rvs(n)[:, :3]
    return partial(flat_plane_to_rn, mtx)


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
    mtx = ortho_group.rvs(n)[:, :3]
    return partial(torus_to_rn, mtx)


# ----------------------------------- cylinder ---------------------------------- #
def cylinder_to_rn(mtx, p):
    """
        Embedd a cylinder by first embedding it in
        R3 and then using a linear transformatin
        to Rn
    """
    cylinder_3d = cylinder_to_r3(p)
    embedded = mtx @ np.array(cylinder_3d)
    return tuple(embedded)


def prepare_cylinder_to_rn(n=64):
    mtx = ortho_group.rvs(n)[:, :3]
    return partial(cylinder_to_rn, mtx)
