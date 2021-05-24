import matplotlib.pyplot as plt

from manifold import embeddings, Sphere, Plane, Torus
from manifold.rnn import RNN
from manifold import vectors_fields

MANIFOLD = "torus"
if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r3, n_sample_points=[3, 2])
    x_range = 0.1
elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[4, 4])
    x_range = 0.05
else:
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[2, 6])
    x_range = 0.05

# set vector field
M.vectors_field = vectors_fields.first_only

# visualize in embedding
ax = M.visualize_embedded()
M.visualize_base_functions_at_point(ax, x_range=[x_range, x_range], scale=0.25)
# M.visualize_charts()


# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=64, scale=0.2)
rnn.run_points(n_seconds=100)
rnn.plot_traces(ax, skip=10)

plt.show()
