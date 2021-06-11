# import numpy as np
# from dataclasses import dataclass
# from typing import Callable
# from functools import partial
from autograd import jacobian, elementwise_grad
from autograd import numpy as np

from manifold.maths import ortho_normal_matrix
from manifold.manifolds import _embeddings

# --------------------------------- wrappers --------------------------------- #


# ---------------------------------------------------------------------------- #
#                                    R^N = 3                                   #
# ---------------------------------------------------------------------------- #


class Embedding:
    def __init__(self, name, phi):
        self.name = name
        self.phi = phi

        self.push_forward = elementwise_grad(phi)

    def __call__(self, *args):
        return self.phi(*args)

    def __repr__(self):
        return f"Embedding map: {self.name}"

    def __str__(self):
        return f"Embedding map: {self.name}"


class TwoStepsEmbedding:
    N = 64

    def __init__(self, name, phi_1, scale=1):
        mtx = ortho_normal_matrix(self.N, 3)
        self.name = name
        self.mtx = mtx
        self.phi_1 = phi_1
        self.scale = scale

        def _embedd(p):
            return mtx @ np.array(phi_1(p)) * scale

        self._push_forward = jacobian(_embedd)

    def _embedd(self, p):
        return self.mtx @ np.array(self.phi_1(p)) * self.scale

    def push_forward(self, x):
        return self._push_forward(x)

    def __call__(self, p):
        return self._embedd(p)

    def __repr__(self):
        return f"Embedding map: {self.name}"

    def __str__(self):
        return f"Embedding map: {self.name}"


# ----------------------------------- line ----------------------------------- #

line_to_r3_flat = Embedding("line to R3 flat", _embeddings.line_to_r3_flat)
line_to_r3 = Embedding("line to R3", _embeddings.line_to_r3)
helix_to_r3 = Embedding("helix to R3", _embeddings.helix_to_r3)

# ---------------------------------- circle ---------------------------------- #
circle_to_r3_flat = Embedding(
    "circle to R3 flat", _embeddings.circle_to_r3_angled
)
circle_to_r3 = Embedding("circle to R3", _embeddings.circle_to_r3_bent)
circle_to_r3_curvy = Embedding("circle to R3 curvy", _embeddings.circle_to_r3)

# ---------------------------------- sphere ---------------------------------- #
sphere_to_r3 = Embedding("sphere to R3", _embeddings.sphere_to_r3)

# ----------------------------------- plane ---------------------------------- #
plane_to_r3_flat = Embedding("plane to R3 flat", _embeddings.plane_to_r3_flat)
plane_to_r3 = Embedding("plane to R3", _embeddings.plane_to_r3)

# ----------------------------------- torus ---------------------------------- #
torus_to_r3 = Embedding("torus to R3", _embeddings.torus_to_r3)

# --------------------------------- cylinder --------------------------------- #
cylinder_to_r3 = Embedding("cylinder to R3", _embeddings.cylinder_to_r3)
cone_to_r3 = Embedding("cone to R3", _embeddings.cylinder_to_r3_as_cone)


# ---------------------------------------------------------------------------- #
#                                    R^N > 3                                   #
# ---------------------------------------------------------------------------- #


# ----------------------------------- line ----------------------------------- #
line_to_rn_flat = TwoStepsEmbedding(
    "line to rn flat", _embeddings.line_to_r3_flat
)
line_to_rn = TwoStepsEmbedding("line to rn", _embeddings.line_to_r3)
helix_to_rn = TwoStepsEmbedding("helix to rn", _embeddings.helix_to_r3)


# ---------------------------------- circle ---------------------------------- #
circle_to_rn_flat = TwoStepsEmbedding(
    "circle to rn flat", _embeddings.circle_to_r3_angled, scale=3
)
circle_to_rn = TwoStepsEmbedding(
    "circle to rn", _embeddings.circle_to_r3, scale=3
)
circle_to_rn_bent = TwoStepsEmbedding(
    "circle to rn bent", _embeddings.circle_to_r3_bent, scale=3
)


# ---------------------------------- sphere ---------------------------------- #
sphere_to_rn = TwoStepsEmbedding("sphere to rn", _embeddings.sphere_to_r3)
ellipse_to_rn = TwoStepsEmbedding("ellipse to rn", _embeddings.ellipse_to_r3)

# ---------------------------------- plane ---------------------------------- #
plane_to_rn = TwoStepsEmbedding("plane to rn", _embeddings.plane_to_r3)
plane_to_rn_flat = TwoStepsEmbedding(
    "plane to rn flat", _embeddings.plane_to_r3_flat, scale=2
)

# ----------------------------------- torus ---------------------------------- #
torus_to_rn = TwoStepsEmbedding("torus to rn", _embeddings.torus_to_r3)
thin_torus_to_rn = TwoStepsEmbedding(
    "thin torus to rn", _embeddings.thin_torus_to_r3
)

# ----------------------------------- cylinder ---------------------------------- #
cylinder_to_rn = TwoStepsEmbedding(
    "cylinder to rn", _embeddings.cylinder_to_r3
)
cone_to_rn = TwoStepsEmbedding(
    "cone to rn", _embeddings.cylinder_to_r3_as_cone
)
