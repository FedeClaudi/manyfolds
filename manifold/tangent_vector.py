import numpy as np
from loguru import logger

from manifold.maths import unit_vector


def take_derivative_at_point(arr, idx):
    """
        Takes the derivative of a function's image
        at a point.
        In practice we get the average of the derivative
        at the point and at the next sample in the function
    """

    derivative = np.mean(np.diff(arr, axis=0)[idx : idx + 1, :], 0)
    if np.linalg.norm(derivative) == 0:
        logger.warning(f"Tangent vector for base function is vanishing")
        # derivative += 1e-10
    else:
        derivative /= np.linalg.norm(derivative)

    return derivative.T


def get_tangent_vector(point, vectors_field=None, debug=False):
    """
        Gets a tangent vector at a point by summing
        the basis vectors using coefficients from a vectors field
    """

    basis = []
    for fn in point.base_functions:
        basis.append(unit_vector(fn.tangent_vector))
    basis = np.vstack(basis)

    if vectors_field is not None:
        weights = vectors_field(point)
    else:
        weights = np.ones(len(basis))

    if debug:
        logger.debug(
            f"Creating tangent vector for point: {point.coordinates} with weights: {weights} and vectors field: {vectors_field}"
        )

    if basis.shape[0] == 1:
        return basis[0] * weights[0]
    else:
        try:
            return np.dot(basis.T, weights)
        except Exception:
            raise ValueError(
                "Failed to get weighted sum of basis vector - likely a vector field for a 1D manifold was used on a 2D manifold"
            )
