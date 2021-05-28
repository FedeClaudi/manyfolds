import numpy as np

from manifold import embeddings, Circle, Visualizer
from manifold.rnn import RNN
from manifold import vectors_fields


M = Circle(embeddings.circle_to_r3_flat, n_sample_points=8)

# define vector field
M.vectors_field = vectors_fields.small


def vfield_one(point):
    return (1,)


def vfield_two(point):
    return (-1,)


K = 24

# create RNN
rnn = RNN(M, n_units=3, n_inputs=2)
rnn.build_W(k=K, scale=0.01)
rnn.build_B(k=K, vector_fields=[vfield_one, vfield_two])

inputs = np.array([0.2, 0.2])
rnn.run_points(n_seconds=0.5, inputs=inputs)

# visualize
viz = Visualizer(M, rnn)
viz.show(x_range=0.07, scale=0.25, rnn_inputs=inputs)


# TODO we are finding nice bases for input vectors and B looks good
# TODO now make this work as a 1D ring attractor
