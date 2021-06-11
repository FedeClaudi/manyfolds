from loguru import logger

from manifold import embeddings, Line, Circle, Torus, Sphere, Plane, Cylinder
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer

# --------------------------------- settings --------------------------------- #
MANIFOLD = "sphere"
N = 64
K = 10

# ---------------------------------------------------------------------------- #
#                          select manifold parameters                          #
# ---------------------------------------------------------------------------- #
# get manifold
if MANIFOLD == "line":
    logger.debug("Line manifold")
    M = Line(embeddings.line_to_rn, n_sample_points=1)
    pca_sample_points = 50

    if K > 6:
        logger.warning("Line manifold prefers K = 6")

elif MANIFOLD == "helix":
    logger.debug("helix manifold")
    M = Line(embeddings.helix_to_rn, n_sample_points=4)
    pca_sample_points = 150

    if K > 12 or K < 8:
        logger.warning("Helix manifold prefers K = 10")

elif MANIFOLD == "circle":
    logger.debug("Circle manifold")
    M = Circle(embeddings.circle_to_rn, n_sample_points=6)
    pca_sample_points = 50
    # M.vectors_field = vectors_fields.double_sin

elif MANIFOLD == "torus":
    logger.debug("Torus manifold")
    M = Torus(embeddings.torus_to_rn, n_sample_points=[8, 4])
    pca_sample_points = 50
    M.vectors_field = vectors_fields.second_only


elif MANIFOLD == "sphere":

    logger.debug("Sphere manifold")
    M = Sphere(embeddings.sphere_to_rn, n_sample_points=[4, 4])
    pca_sample_points = 75

    M.vectors_field = vectors_fields.sphere_base

    if K != 12:
        logger.warning("Sphere manifold prefers K = 12")

elif MANIFOLD == "cylinder":
    logger.debug("Cylinder manifold")
    M = Cylinder(
        embeddings.prepare_cylinder_to_rn(n=N), n_sample_points=[3, 4]
    )
    pca_sample_points = 60

    M.vectors_field = vectors_fields.cylinder_vfield


elif MANIFOLD == "cone":
    logger.debug("Cylinder manifold")
    M = Cylinder(
        embeddings.prepare_cylinder_as_cone_to_rn(n=N), n_sample_points=[3, 4]
    )
    pca_sample_points = 60

    M.vectors_field = vectors_fields.first_only

elif MANIFOLD == "plane":
    logger.debug("Plane manifold")
    M = Plane(embeddings.plane_to_rn_flat, n_sample_points=[2, 2])
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
rnn.run_points(n_seconds=60, cut=True)

viz = Visualizer(M, rnn=rnn, pca_sample_points=pca_sample_points)
viz.show(scale=0.3)
