import matplotlib.pyplot as plt

from manifold import embeddings, Sphere, Plane, Torus
from manifold.rnn import RNN
from manifold import vectors_fields

MANIFOLD = "torus"
if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r3, n_sample_points=[3, 2])
    x_range = 0.1
elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[5, 6])
    x_range = 0.2
else:
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 8])
    x_range = 0.2

# create base functions at each point in the manifold
M.get_base_functions()

# visualize in embedding
ax = M.visualize_embedded()
M.visualize_base_functions_at_point(
    ax, x_range=[x_range * 2, x_range], scale=0
)
# M.visualize_charts()

# set vector field
M.vectors_field = vectors_fields.identity

# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=[20, 20], scale=1)
rnn.run_points(n_seconds=10)
rnn.plot_traces(ax, skip=10)

plt.show()
