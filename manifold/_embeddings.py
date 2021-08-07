import numpy as np
from dataclasses import dataclass
from typing import Callable

from manifold.maths import ortho_normal_matrix


"""
    Basis claasses to create embeddings in R3 (Embedding) or RN (TwoStepsEmbedding)
"""


@dataclass
class Embedding:
    """
        Base embedding class for embeddings in three dimensions
    """

    name: str
    phi: Callable

    def __call__(self, *args):
        return self.phi(*args)


class TwoStepsEmbedding:
    def __init__(self, name, phi_1, scale=1, N=64):
        """
            Base embedding class for embedding in R^n.
            The embedding is done in two steps: phi_1 embeds onto
            R^3 and an orthonormal matrix does R^3 -> R^n.
        """
        self.N = N
        self.name = name
        self.mtx = ortho_normal_matrix(self.N, 3)
        self.phi_1 = phi_1
        self.scale = scale

    def __call__(self, p):
        embedded = self.mtx @ np.array(self.phi_1(p)) * self.scale
        return embedded
