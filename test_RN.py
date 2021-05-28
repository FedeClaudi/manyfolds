from loguru import logger

from manifold import embeddings, Line, Circle, Torus, Sphere, Plane, Cylinder
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer

MANIFOLD = "circle"
N = 12
K = 6

# get manifold
if MANIFOLD == "line":
    logger.debug("Line manifold")
    M = Line(embeddings.prepare_line_to_rn(n=N), n_sample_points=1)
    pca_sample_points = 50

elif MANIFOLD == "helix":
    logger.debug("helix manifold")
    M = Line(embeddings.prepare_helix_to_rn(n=N), n_sample_points=4)
    pca_sample_points = 150

elif MANIFOLD == "circle":
    logger.debug("Circle manifold")
    M = Circle(embeddings.prepare_circle_to_rn(n=N), n_sample_points=10)
    pca_sample_points = 100


elif MANIFOLD == "torus":
    logger.debug("Torus manifold")
    M = Torus(embeddings.prepare_torus_to_rn(n=N), n_sample_points=[4, 2])
    pca_sample_points = 60

    M.vectors_field = vectors_fields.second_only


elif MANIFOLD == "sphere":
    logger.debug("Sphere manifold")
    M = Sphere(embeddings.prepare_sphere_to_rn(n=N), n_sample_points=[4, 4])
    pca_sample_points = 60

    M.vectors_field = vectors_fields.second_only

elif MANIFOLD == "cylinder":
    logger.debug("Cylinder manifold")
    M = Cylinder(
        embeddings.prepare_cylinder_to_rn(n=N), n_sample_points=[4, 4]
    )
    pca_sample_points = 20

elif MANIFOLD == "plane":
    logger.debug("Plane manifold")
    M = Plane(embeddings.prepare_plane_to_rn(n=N), n_sample_points=[2, 2])
    pca_sample_points = 20

else:
    raise NotImplementedError

M.print_embedding_bounds()

# set vector field
# M.vectors_field = vectors_fields.sphere_equator

# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=0.001)
rnn.run_points(n_seconds=5)


viz = Visualizer(M, rnn=rnn, pca_sample_points=pca_sample_points)
viz.show(x_range=[0.1, 0.2], scale=0.2)
