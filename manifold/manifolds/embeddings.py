from manifold.manifolds._embeddings import Embedding, Embedding2D

# # ---------------------------------------------------------------------------- #
# #                                    R^N = 3                                   #
# # ---------------------------------------------------------------------------- #

# ----------------------------------- line ----------------------------------- #
line_to_r3_flat = Embedding("line_to_r3_flat", "p, p, p")
line_to_r3 = Embedding(
    "line_to_r3", "sin(2 * p) - 0.5, sin(p) * 2 - 1, -cos(p) * 4 + 3"
)
helix_to_r3 = Embedding(
    "helix_to_r3", "cos(4 * pi * p) / 2, sin(4 * pi * p) / 2, p + 0.25"
)

# ---------------------------------- circle ---------------------------------- #
circle_to_r3_angled = Embedding(
    "circle_to_r3_angled", "sin(p), cos(p), sin(p)"
)
circle_to_r3_bent = Embedding(
    "circle_to_r3_bent", "sin(p), 0.8 * cos(p), cos(p) ** 2 * 0.5 + 0.5"
)
circle_to_r3 = Embedding(
    "circle_to_r3", "sin(p), 0.8 * cos(p), cos(p * 2) ** 2 * 0.5 + 0.5"
)


# ----------------------------------- plane ---------------------------------- #

plane_to_r3_flat = Embedding2D(
    "plane_to_r3_flat", "p0 + 0.2, p1 + 0.2, 0.5 * (p0 + p1)"
)
plane_to_r3 = Embedding2D(
    "plane_to_r3", "p0, sin(p1) * 2, 0.4 * (p1 - p0) ** 2"
)


# ---------------------------------- sphere ---------------------------------- #
sphere_to_r3 = Embedding2D(
    "sphere_to_r3", "sin(p0) * cos(p1), sin(p0) * sin(p1), cos(p0)"
)

# ----------------------------------- torus ---------------------------------- #
torus_to_r3 = Embedding2D(
    "torus_to_r3",
    "(.75 + .25 * cos(p0)) * cos(p1), (.75+ .25 * cos(p0)) * sin(p1), .25 * sin(p0)",
)


# def torus_to_r3(p0, p1):
#     R = 0.75  # torus center -> tube center
#     r = 0.25  # tube radius
#     return (
#         (R + r * cos(p0)) * cos(p1),
#         (R + r * cos(p0)) * sin(p1),
#         r * sin(p0),
#     )

# --------------------------------- cylinder --------------------------------- #
cylinder_to_r3 = Embedding2D(
    "cylinder_to_r3", "sin(p0) / 2, cos(p0) / 2, p1 + 0.1"
)

cylinder_to_r3_as_cone = Embedding2D(
    "cylinder_to_r3_as_cone",
    "p1 / 2 + 0.4 * sin(p0) / 2, p1 / 2 + 0.4 * cos(p0) / 2, p1 + 0.5",
)
