import numpy as np
from dataclasses import dataclass
from typing import Callable

from manifold.maths import ortho_normal_matrix
from manifold.manifolds import _embeddings

# --------------------------------- wrappers --------------------------------- #


# ---------------------------------------------------------------------------- #
#                                    R^N = 3                                   #
# ---------------------------------------------------------------------------- #


@dataclass
class Embedding:
    name: str
    phi: Callable

    def __call__(self, *args):
        return self.phi(*args)


class TwoStepsEmbedding:
    def __init__(self, name, phi_1, scale=1, N=64):
        self.N = N
        self.name = name
        self.mtx = ortho_normal_matrix(self.N, 3)
        self.phi_1 = phi_1
        self.scale = scale

    def __call__(self, p):
        embedded = self.mtx @ np.array(self.phi_1(p)) * self.scale
        return embedded


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
circle_to_r3_bent = Embedding(
    "circle to R3 curvy", _embeddings.circle_to_r3_bent
)

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
    "circle to rn flat", _embeddings.circle_to_r3_angled, scale=2
)
circle_to_rn = TwoStepsEmbedding(
    "circle to rn", _embeddings.circle_to_r3, scale=2
)
circle_to_rn_bent = TwoStepsEmbedding(
    "circle to rn bent", _embeddings.circle_to_r3_bent, scale=2
)


# ---------------------------------- sphere ---------------------------------- #
sphere_to_rn = TwoStepsEmbedding(
    "sphere to rn", _embeddings.sphere_to_r3, scale=1
)
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
