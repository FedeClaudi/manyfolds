import numpy as np

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
    return (np.sin(point[0] * 2),)


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
@normalize
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
    return (np.cos(point.coordinates[0]), 0)


@sel_args
@normalize
def sphere_poles(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (-np.cos(point.coordinates[0]), 0)


@sel_args
@normalize
def sphere_base(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (-np.cos(point.coordinates[0]), 1)


# ----------------------------------- torus ---------------------------------- #
@normalize
def torus_base(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (point.coordinates[0], np.sin(2 * point.coordinates[1]))


@sel_args
def torus_first(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (point.coordinates[0], 0)


@sel_args
def torus_second(point):
    """
        Pushes the dynamics towards the poles of the sphere
    """
    return (0, np.sin(2 * point.coordinates[1]))
