from loguru import logger
from vedo import screenshot

import sys

sys.path.append("./")
from manifold import embeddings, Line, Circle, Torus, Sphere, Plane, Cylinder
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer
from manifold import visualize

visualize.rnn_trace_alpha = 0.6
visualize.reco_surface_radius = 0.1

# --------------------------------- settings --------------------------------- #
MANIFOLD = "cylinder"
N = 64
K = 12

# ---------------------------------------------------------------------------- #
#                          select manifold parameters                          #
# ---------------------------------------------------------------------------- #
cam = None
# get manifold
if MANIFOLD == "line":
    logger.debug("Line manifold")
    M = Line(embeddings.line_to_rn, n_sample_points=1)
    pca_sample_points = 50

    if K > 6:
        logger.warning("Line manifold prefers K = 6")

    cam = dict(
        pos=(-1.05, 0.545, 5.27),
        focalPoint=(0.0378, 0.0910, 5.63e-3),
        viewup=(0.756, 0.646, 0.101),
        distance=5.39,
        clippingRange=(4.62, 6.38),
    )

elif MANIFOLD == "helix":
    logger.debug("helix manifold")
    M = Line(embeddings.helix_to_rn, n_sample_points=4)
    pca_sample_points = 150

    if K > 12 or K < 8:
        logger.warning("Helix manifold prefers K = 10")

elif MANIFOLD == "circle":
    logger.debug("Circle manifold")
    # M = Circle(embeddings.circle_to_r3_flat, n_sample_points=2)
    M = Circle(embeddings.circle_to_r3, n_sample_points=2)

    pca_sample_points = 50
    # M.vectors_field = vectors_fields.double_sin

    cam = dict(
        pos=(-5.18, 5.94, 1.54),
        focalPoint=(-0.122, -0.0158, 0.0684),
        viewup=(0.154, -0.112, 0.982),
        distance=7.95,
        clippingRange=(5.30, 11.5),
    )

elif MANIFOLD == "torus":
    logger.debug("Torus manifold")
    M = Torus(embeddings.torus_to_rn, n_sample_points=[6, 5])
    pca_sample_points = 100
    M.vectors_field = vectors_fields.second_only

    cam = dict(
        pos=(0.172, -5.61, 1.11),
        focalPoint=(-2.15e-3, 0.0103, -3.00e-5),
        viewup=(7.81e-3, 0.194, 0.981),
        distance=5.73,
        clippingRange=(3.51, 8.53),
    )

elif MANIFOLD == "sphere":

    logger.debug("Sphere manifold")
    M = Sphere(embeddings.sphere_to_rn, n_sample_points=[6, 6])
    pca_sample_points = 75

    M.vectors_field = vectors_fields.second_only

    if K != 12:
        logger.warning("Sphere manifold prefers K = 12")

    cam = dict(
        pos=(-0.772, 2.74, 6.26),
        focalPoint=(2.77e-4, 8.17e-3, 2.33e-6),
        viewup=(-0.994, -0.0371, -0.106),
        distance=6.87,
        clippingRange=(3.88, 10.7),
    )

elif MANIFOLD == "cylinder":
    logger.debug("Cylinder manifold")
    M = Cylinder(embeddings.cylinder_to_rn, n_sample_points=[3, 4])
    pca_sample_points = 60

    M.vectors_field = vectors_fields.first_only
    cam = dict(
        pos=(-0.227, -3.68, 0.366),
        focalPoint=(-1.16e-3, 9.88e-3, 0),
        viewup=(-0.0146, 0.0997, 0.995),
        distance=3.71,
        clippingRange=(2.43, 5.34),
    )

elif MANIFOLD == "cone":
    logger.debug("Cylinder manifold")
    M = Cylinder(embeddings.cone_to_rn, n_sample_points=[3, 4])
    pca_sample_points = 60

    M.vectors_field = vectors_fields.first_only

elif MANIFOLD == "plane":
    logger.debug("Plane manifold")
    M = Plane(embeddings.plane_to_rn, n_sample_points=[2, 2])
    pca_sample_points = 80

    # M.vectors_field = vectors_fields.second_only

else:
    raise NotImplementedError


# ---------------------------------------------------------------------------- #
#                              stuff happens here                              #
# ---------------------------------------------------------------------------- #
M.print_embedding_bounds()

# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=15, cut=True)


viz = Visualizer(
    M, rnn=rnn, manifold_alpha=1, pca_sample_points=pca_sample_points
)
viz.show(x_range=[0.1, 0.2], scale=0.2, axes=0, camera=cam)

screenshot(f"paper/images/3A_{MANIFOLD}.png")
