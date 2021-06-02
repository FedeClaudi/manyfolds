import numpy as np
import itertools
from loguru import logger


def gram_schmidt(vectors):
    basis = []
    for v in vectors.T:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return np.array(basis).T


def ortho_normal_matrix(n, m):
    X = np.random.randn(n, m)
    return gram_schmidt(X)


def ReLU(_, x):
    x = np.array(x)
    return x * (x > 0)


# def tanh(_, x):
#     x = np.array(x)
#     return np.tanh(x)


def cartesian_product(X, Y):
    return np.array(list(itertools.product(X, Y)))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norm = np.linalg.norm(vector)
    if not norm:
        logger.warning("Attempted to normalize a vector with no magnitude")
        return vector
    else:
        return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def min_distance_from_point(vec, p):
    """
        Minimial distance between a single point and each point along a vector (in N dimensions)
    """
    return np.apply_along_axis(np.linalg.norm, 1, vec - p).min()
