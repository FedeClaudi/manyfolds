import numpy as np

# from manifold import embeddings, Line, Circle, Torus, Sphere, Plane, Cylinder
from manifold import Line, embeddings, Circle
from manifold.rnn import RNN

# from manifold import vectors_fields
from manifold import Visualizer

N = 64
K1 = 6
K2 = 12

# ---------------------------- build two manifolds --------------------------- #
M1 = Line(embeddings.prepare_line_to_rn(n=N), n_sample_points=4)
M2 = Circle(embeddings.prepare_circle_angled_to_rn(n=N), n_sample_points=6)


# ------------------------------- fit two  RNNs ------------------------------ #
rnn1 = RNN(M1, n_units=N)
rnn1.build_W(k=K1, scale=1)

rnn2 = RNN(M2, n_units=N)
rnn2.build_W(k=K2, scale=1)

# ---------------------------------- combine --------------------------------- #
W = np.zeros((2 * N, 2 * N))
W[:N, :N] = rnn1.W
W[N:, N:] = rnn2.W


# combine embedded points
h = np.hstack([M1.points[0].embedded, M2.points[0].embedded])

# run a new RNN
rnn = RNN()
rnn.W = W
rnn.run_initial_condition(h, n_seconds=50)


# visualize
viz = Visualizer(manifold=None, rnn=rnn)
viz.show()
