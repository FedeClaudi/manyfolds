import matplotlib.pyplot as plt

from manifold import embeddings, Sphere, Plane

# from manifold.rnn import RNN

# from manifold.maths import angle_between, min_distance_from_point

MANIFOLD = "sphere"
if MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_r3, n_sample_points=[4, 4])
else:
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[3, 3])

# create base functions at each point in the manifold
M.get_base_functions()

# create RNN
# rnn = RNN(M, n_units=3)
# rnn.build_W(k=64, scale=10)
# rnn.run_points(n_seconds=50)

# visualize in embedding
ax = M.visualize_embedded()
M.visualize_base_functions_at_point(ax, x_range=[0.1, 0.3], scale=0.3)

# visualize RNN dynamics
# rnn.plot_traces(ax)

plt.show()
