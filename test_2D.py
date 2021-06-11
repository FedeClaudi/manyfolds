from manifold import embeddings, Sphere, Plane, Torus, Cylinder
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer

K = 12
MANIFOLD = "sphere"

if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r3, n_sample_points=[3, 2])
    x_range = [0.3, 0.3]

    M.vectors_field = vectors_fields.second_only

elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[8, 0])
    x_range = [0.1, 0.05]
    M.vectors_field = vectors_fields.second_only

elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 10])
    x_range = [0.02, 0.02]
    M.vectors_field = vectors_fields.second_only

elif MANIFOLD == "cylinder":
    M = Cylinder(embeddings.cylinder_to_r3, n_sample_points=[6, 2])
    M.vectors_field = vectors_fields.cylinder_vfield
    x_range = [0.1, 0.05]

M.print_embedding_bounds()

# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=10)


# visualize in embedding
viz = Visualizer(M, None)

for point in M.points:
    for fn in point.base_functions:
        fn_coords = fn.manifold_coords[50, :]
        p = point.embedding_map.push_forward(fn_coords)
        a = 1

    viz.visualize_basis_vectors_at_point(point, scale=0.15)

viz.show(x_range=0.2, scale=0.27)
