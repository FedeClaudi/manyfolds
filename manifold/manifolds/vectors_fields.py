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
@normalize
def identity(point):
    return tuple([1 for n in range(point.d)])


# ---------------------------------------------------------------------------- #
#                                      1D                                      #
# ---------------------------------------------------------------------------- #


@sel_args
def sin(point):
    return (np.sin(point[0]),)


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
def scale_first(point, fact=50):
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
