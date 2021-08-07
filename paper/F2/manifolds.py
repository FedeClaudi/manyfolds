from loguru import logger
import sys
from vedo import screenshot

sys.path.append("./")
from manifold import embeddings, Line, Circle, Sphere, Plane, Cylinder
from manifold import Visualizer


"""
    Plots different manifolds with tangent vectors on them.
    Use MANIFOLDD to change which manifold to show and different embeddings maps can
    be used.
"""

# --------------------------------- settings --------------------------------- #
MANIFOLD = "sphere"  # select which manifold to show
N = 64  # number of dimensions/RNN units

# ---------------------------------------------------------------------------- #
#                          select manifold parameters                          #
# ---------------------------------------------------------------------------- #
# get manifold
if MANIFOLD == "line":
    logger.debug("Line manifold")
    M = Line(embeddings.line_to_rn, n_sample_points=3)


elif MANIFOLD == "helix":
    logger.debug("helix manifold")
    M = Line(embeddings.helix_to_rn, n_sample_points=3)


elif MANIFOLD == "circle":
    logger.debug("Circle manifold")
    M = Circle(embeddings.circle_to_rn, n_sample_points=3)

elif MANIFOLD == "sphere":
    logger.debug("Sphere manifold")
    M = Sphere(embeddings.sphere_to_rn, n_sample_points=[0, 0])

elif MANIFOLD == "cylinder":
    logger.debug("Cylinder manifold")
    M = Cylinder(
        embeddings.prepare_cylinder_to_rn(n=N), n_sample_points=[1, 0]
    )

elif MANIFOLD == "cone":
    logger.debug("Cylinder manifold")
    M = Cylinder(
        embeddings.prepare_cylinder_as_cone_to_rn(n=N), n_sample_points=[1, 0]
    )

elif MANIFOLD == "plane":
    logger.debug("Plane manifold")
    M = Plane(embeddings.plane_to_rn, n_sample_points=[1, 0])
else:
    raise NotImplementedError


# ---------------------------------------------------------------------------- #
#                              stuff happens here                              #
# ---------------------------------------------------------------------------- #


viz = Visualizer(M, pca_sample_points=100, axes=0)
viz.show(scale=0.3)

screenshot(f"paper/images/F2C_{MANIFOLD}.png")
