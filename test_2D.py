import matplotlib.pyplot as plt

from manifold import embeddings, Sphere, Plane, Torus, Cylinder
from manifold.rnn import RNN

# from manifold import vectors_fields

MANIFOLD = "sphere"
if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r3, n_sample_points=[3, 2])
    x_range = [0.1, 0.1]
elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[4, 4])
    x_range = [0.1, 0.05]
elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[2, 6])
    x_range = [0.1, 0.05]
elif MANIFOLD == "cylinder":
    M = Cylinder(embeddings.cylinder_to_r3_as_cone, n_sample_points=[6, 2])
    x_range = [0.1, 0.05]

# set vector field
# M.vectors_field = vectors_fields.first_only

# visualize in embedding
ax = M.visualize_embedded()
M.visualize_base_functions_at_point(ax, x_range=x_range, scale=0.005)
# M.visualize_charts()


# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=6, scale=0.02)
rnn.run_points(n_seconds=5)
rnn.plot_traces(ax, skip=10)

plt.show()
