import matplotlib.pyplot as plt

from manifold import embeddings, Sphere, Plane, Torus
from manifold.rnn import RNN


MANIFOLD = "sphere"
if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r3, n_sample_points=[8, 8])
    x_range = 0.1
elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[8, 8])
    x_range = 0.4
else:
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 8])
    x_range = 0.2

# create base functions at each point in the manifold
M.get_base_functions()

# visualize in embedding
ax = M.visualize_embedded()
M.visualize_base_functions_at_point(ax, x_range=[x_range, x_range], scale=0.03)
# M.visualize_charts()

# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=64, scale=10)
rnn.run_points(n_seconds=50)
rnn.plot_traces(ax)

plt.show()
