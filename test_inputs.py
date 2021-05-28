import numpy as np

from manifold import embeddings, Circle, Visualizer, Sphere
from manifold.rnn import RNN
from manifold import vectors_fields

from functools import partial


K = 24
n_inputs = 10
MANIFOLD = "circle"

if MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_r3_flat, n_sample_points=5)
elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 3])

# define vector field
M.vectors_field = vectors_fields.small


# def vfield_one(point):
#     return (np.sin(point[0]),)


# def vfield_one(point):
#     return (np.sin(point[0]),)


# def vfield_two(point):
#     return (-np.sin(point[0] + 0.1),)


def vec_field(x, y, point):
    return (x, y)


vec_fileds = []
X = np.linspace(-1, 1, n_inputs)
Y = np.linspace(1, -1, n_inputs)
for x, y in zip(X, Y):
    vec_fileds.append(partial(vec_field, x, y))

# create RNN
rnn = RNN(M, n_units=3, n_inputs=n_inputs)
rnn.build_W(k=K, scale=0.01)
rnn.build_B(k=K, vector_fields=vec_fileds)

inputs = np.ones(n_inputs) * 0.5
rnn.run_points(n_seconds=0.5, inputs=inputs)

# visualize
viz = Visualizer(M, rnn)
viz.show(x_range=0.07, scale=0.5, rnn_inputs=inputs)


# TODO we are finding nice bases for input vectors and B looks good
# TODO now make this work as a 1D ring attractor
