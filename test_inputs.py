import numpy as np
from numpy import pi

from manifold import embeddings, Circle, Visualizer, Sphere
from manifold.rnn import RNN
from manifold import vectors_fields

from functools import partial


K = 24
n_inputs = 8
MANIFOLD = "circle"

if MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_r3_flat, n_sample_points=5)
elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 3])

# define vector field
M.vectors_field = vectors_fields.small



def vfield(sign, shift, point):
    val = np.sin(point[0] + shift) * sign
    return (val, )

shifts = [0, pi*.25, pi*5, pi*.75]
vfields_pos, vfields_neg = [], []
for shift in shifts:
    vfields_pos.append(partial(vfield, 1, shift))
    vfields_neg.append(partial(vfield, -1, shift))
vfields = vfields_pos + vfields_neg


# create RNN

rnn = RNN(M, n_units=3, n_inputs=n_inputs)
rnn.build_W(k=K, scale=0.01)
rnn.build_B(k=K, vector_fields=vfields)

# inputs = np.random.uniform(-1, 1, size=(20, 2))
# inputs[0, :] = 1
# inputs[0, :] = -1
# inputs = np.random.uniform(-1, 1, size=(2, n_inputs))

rnn.run_points(n_seconds=0.5, inputs=None)

# visualize
viz = Visualizer(M, rnn)
viz.show(x_range=0.07, scale=1, rnn_inputs=None)


# TODO we are finding nice bases for input vectors and B looks good
# TODO now make this work as a 1D ring attractor
