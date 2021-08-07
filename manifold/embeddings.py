from manifold import embedding_functions

from manifold._embeddings import Embedding, TwoStepsEmbedding

"""
    Embeddings of various one and two dimensional manifolds into
    R^3 and R^64
"""

# ---------------------------------------------------------------------------- #
#                                    R^N = 3                                   #
# ---------------------------------------------------------------------------- #

# ----------------------------------- line ----------------------------------- #

line_to_r3_flat = Embedding(
    "line to R3 flat", embedding_functions.line_to_r3_flat
)
line_to_r3 = Embedding("line to R3", embedding_functions.line_to_r3)
helix_to_r3 = Embedding("helix to R3", embedding_functions.helix_to_r3)

# ---------------------------------- circle ---------------------------------- #
circle_to_r3_flat = Embedding(
    "circle to R3 flat", embedding_functions.circle_to_r3_angled
)
circle_to_r3 = Embedding("circle to R3", embedding_functions.circle_to_r3_bent)
circle_to_r3_curvy = Embedding(
    "circle to R3 curvy", embedding_functions.circle_to_r3
)
circle_to_r3_bent = Embedding(
    "circle to R3 curvy", embedding_functions.circle_to_r3_bent
)

# ---------------------------------- sphere ---------------------------------- #
sphere_to_r3 = Embedding("sphere to R3", embedding_functions.sphere_to_r3)

# ----------------------------------- plane ---------------------------------- #
plane_to_r3_flat = Embedding(
    "plane to R3 flat", embedding_functions.plane_to_r3_flat
)
plane_to_r3 = Embedding("plane to R3", embedding_functions.plane_to_r3)

# ----------------------------------- torus ---------------------------------- #
torus_to_r3 = Embedding("torus to R3", embedding_functions.torus_to_r3)

# --------------------------------- cylinder --------------------------------- #
cylinder_to_r3 = Embedding(
    "cylinder to R3", embedding_functions.cylinder_to_r3
)
cone_to_r3 = Embedding(
    "cone to R3", embedding_functions.cylinder_to_r3_as_cone
)


# ---------------------------------------------------------------------------- #
#                                    R^N > 3                                   #
# ---------------------------------------------------------------------------- #


# ----------------------------------- line ----------------------------------- #
line_to_rn_flat = TwoStepsEmbedding(
    "line to rn flat", embedding_functions.line_to_r3_flat
)
line_to_rn = TwoStepsEmbedding("line to rn", embedding_functions.line_to_r3)
helix_to_rn = TwoStepsEmbedding("helix to rn", embedding_functions.helix_to_r3)


# ---------------------------------- circle ---------------------------------- #
circle_to_rn_flat = TwoStepsEmbedding(
    "circle to rn flat", embedding_functions.circle_to_r3_angled, scale=2
)
circle_to_rn = TwoStepsEmbedding(
    "circle to rn", embedding_functions.circle_to_r3, scale=2
)
circle_to_rn_bent = TwoStepsEmbedding(
    "circle to rn bent", embedding_functions.circle_to_r3_bent, scale=2
)


# ---------------------------------- sphere ---------------------------------- #
sphere_to_rn = TwoStepsEmbedding(
    "sphere to rn", embedding_functions.sphere_to_r3, scale=1
)
ellipse_to_rn = TwoStepsEmbedding(
    "ellipse to rn", embedding_functions.ellipse_to_r3
)

# ---------------------------------- plane ---------------------------------- #
plane_to_rn = TwoStepsEmbedding("plane to rn", embedding_functions.plane_to_r3)
plane_to_rn_flat = TwoStepsEmbedding(
    "plane to rn flat", embedding_functions.plane_to_r3_flat, scale=2
)

# ----------------------------------- torus ---------------------------------- #
torus_to_rn = TwoStepsEmbedding("torus to rn", embedding_functions.torus_to_r3)
thin_torus_to_rn = TwoStepsEmbedding(
    "thin torus to rn", embedding_functions.thin_torus_to_r3
)

# ----------------------------------- cylinder ---------------------------------- #
cylinder_to_rn = TwoStepsEmbedding(
    "cylinder to rn", embedding_functions.cylinder_to_r3
)
cone_to_rn = TwoStepsEmbedding(
    "cone to rn", embedding_functions.cylinder_to_r3_as_cone
)
