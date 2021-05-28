from manifold import embeddings, Sphere, Plane, Torus, Cylinder
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer

K = 32
MANIFOLD = "plane"

if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r, n_sample_points=[3, 2])
    x_range = [0.3, 0.3]

elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[8, 0])
    x_range = [0.1, 0.05]
    M.vectors_field = vectors_fields.scale_second

elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 3])
    x_range = [0.02, 0.02]
    M.vectors_field = vectors_fields.sphere_equator

elif MANIFOLD == "cylinder":
    M = Cylinder(embeddings.cylinder_to_r3_as_cone, n_sample_points=[6, 2])
    x_range = [0.1, 0.05]

# set vector field
M.vectors_field = vectors_fields.first_only
#

# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=K, scale=0.001)
rnn.run_points(n_seconds=1)


# visualize in embedding
viz = Visualizer(M, rnn)
viz.show(x_range=0.2, scale=0.25)
