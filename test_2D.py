from loguru import logger

from manifold import embeddings, Sphere, Plane, Torus, Cylinder
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer

K = 12
MANIFOLD = "cylinder"

if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r3, n_sample_points=[3, 2])
    # M.vectors_field = vectors_fields.second_only

elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[8, 0])
    M.vectors_field = vectors_fields.second_only

elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 10])
    M.vectors_field = vectors_fields.sphere_equator

elif MANIFOLD == "cylinder":
    M = Cylinder(embeddings.cylinder_to_r3, n_sample_points=[6, 2])
    M.vectors_field = vectors_fields.second_only

logger.debug(M.embedding)
M.print_embedding_bounds()

# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=10)


# visualize in embedding
viz = Visualizer(M, None)

# from vedo import Tube
# import numpy as np
for point in M.points:
    viz.visualize_basis_vectors_at_point(point, scale=0.15)

    # f1 = np.zeros((100, 2))
    # f1[:, 0] = np.linspace(point[0]-.5, point[0]+.5, 100)
    # f1[:, 1] = point[1]

    # f = [point.embedding_map(f1[i, :]) for i in range(100)]
    # viz.actors.append(Tube(f, r=.02))

viz.show(scale=0.27)
