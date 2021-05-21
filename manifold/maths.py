import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
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
