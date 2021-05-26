import numpy as np
from loguru import logger

# from manifold.manifolds.embeddings import TwoStepsEmbedding


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
        derivative += 0.001
    derivative /= np.linalg.norm(derivative)

    return derivative.T


def get_basis_tangent_vector(point, base_function, embedding):
    return take_derivative_at_point(
        base_function.embedded, base_function.embedded_point_index
    )
    # if not isinstance(embedding, TwoStepsEmbedding):
    #     # Simply get the embedding of the base function and take the derivative
    #     return take_derivative_at_point(base_function.embedded, base_function.embedded_point_index)
    # else:
    #     # do the first embedding step
    #     manifold_coords = base_function.get_manifold_coordinates(x_range=.2)
    #     first_embedd = np.apply_along_axis(embedding.phi_1, 1, manifold_coords)

    #     # take the derivative in first embedding space
    #     vec = take_derivative_at_point(first_embedd, base_function.embedded_point_index)

    #     # push forward to second embedding space
    #     return embedding.mtx @ vec


def get_tangent_vector(point, vectors_field, debug=False):
    """
        Gets a tangent vector at a point by summing
        the basis vectors using coefficients from a vectors field
    """
    weights = vectors_field(point)
    basis = np.vstack([fn.tangent_vector for fn in point.base_functions])

    if debug:
        logger.debug(
            f"Creating tangent vector for point: {point.coordinates} with weights: {weights} and vectors field: {vectors_field}"
        )

    if basis.shape[0] == 1:
        return basis[0] * weights[0]
    else:
        return np.dot(basis.T, weights)
