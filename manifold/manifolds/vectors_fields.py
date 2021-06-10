import numpy as np
from numpy import pi

"""
    Maps:
        psi: M \to R^d
            p \to (a^i_p)

        assigning for each point on the manifold a d-dimensional vector of 
        weights used for the linear combination of basis for the tangent 
        vector space in the embedding
"""


def normalize(func):
    def inner(*args):
        out = func(*args)
        out = np.array(out) / np.linalg.norm(out)
        return out

    return inner


def sel_args(func):
    def inner(*args):
        if len(args) == 1:
            x = args[0]
        else:
            x = args[1]
        return func(x)

    return inner


@sel_args
def identity(point):
    return tuple([1 for n in range(point.d)])


@sel_args
def small_identity(point):
    return tuple([0.5 for n in range(point.d)])


@sel_args
def negative(point):
    return tuple([-1 for n in range(point.d)])


@sel_args
def zeros(point):
    return tuple([1e-6 for n in range(point.d)])


@sel_args
def small(point):
    return tuple([1e-1 for n in range(point.d)])


@sel_args
def random(point):
    return tuple(np.random.rand(point.d))


# ---------------------------------------------------------------------------- #
#                                      1D                                      #
# ---------------------------------------------------------------------------- #


@sel_args
def sin(point):
    return (np.sin(point[0] * 2 * pi),)


@sel_args
def cos(point):
    return (np.cos(point[0] * 2),)


@sel_args
def double_sin(point):
    return (np.sin(point[0] * 4),)


@sel_args
def double_cos(point):
    return (np.cos(point[0] * 4),)


@sel_args
def scaled_sin(point):
    return (np.sin(point[0] * 6),)


# ---------------------------------------------------------------------------- #
#                                      2D                                      #
# ---------------------------------------------------------------------------- #
@sel_args
@normalize
def first_only(point):
    return (1, 0)


@sel_args
@normalize
def second_only(point):
    return (0, 1)


@sel_args
@normalize
def scale_first(point, fact=5):
    """
        Scales the second dimension wrt the first
    """
    return (fact, 1)


@sel_args
@normalize
def scale_second(point, fact=10):
    """
        Scales the second dimension wrt the first
    """
    return (1, fact)


@sel_args
def sin_on_sphere(point):
    return (np.sin(2 * point[0]), 0)


# ---------------------------------------------------------------------------- #
#                                    custom                                    #
# ---------------------------------------------------------------------------- #
# ---------------------------------- sphere ---------------------------------- #
@sel_args
def sphere_equator(point):
    """
        Pushes the dynamics towards the equator of the sphere
    """
    cos = np.cos(point.coordinates[0])
    cos2 = np.cos(2 * point.coordinates[0])
    return (cos * 0.8, (1 - cos2) * 0.4)


@sel_args
def sphere_poles(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (-np.cos(point.coordinates[0]) * 0.8, 0)


@sel_args
def sphere_base(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (-np.cos(point.coordinates[0]) * 0.5, 0.25)


@sel_args
def sphere_base2(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (0.25, -np.cos(2 * point.coordinates[1]))


# ----------------------------------- torus ---------------------------------- #
def torus_base(point):
    return (point[0], np.sin(2 * point[1]))


def torus_first(point):
    return (np.cos(point[1]) * 2, 0)


def torus_second(point):
    return (0, np.cos(point[0] * 2))


# @sel_args
# def torus_first(point):
#     return (np.sin(2 * point[0]), 0)


# @sel_args
# def torus_second(point):
#     return (0, np.sin(2 * point[1]))

# --------------------------------- cylinder --------------------------------- #
def cylinder_vfield(point):
    return (np.sin(point[1] * pi * 0.5) + 0.1, 0)
