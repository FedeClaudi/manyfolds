import sys

sys.path.append("./")

import numpy as np

from manifold import embeddings, Sphere
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer
from manifold import visualize

visualize.point_size = 0.02
visualize.rnn_trace_radius = 0.01

N = 64
K = 12

# M = Circle(embeddings.prepare_circle_angled_to_rn(n=N), n_sample_points=5)
# M.vectors_field = vectors_fields.small_identity
M = Sphere(embeddings.prepare_sphere_to_rn(n=N), n_sample_points=[5, 0])
M.vectors_field = vectors_fields.sphere_base
pca_sample_points = 50


# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)

# add noise to points
point = M.points[3]
points = []

# for point in M.points[1:-1]:
for i in range(10):
    pt = point.clone()
    pt.embedded = np.array(pt.embedded)
    pt.embedded += np.random.randn(*pt.embedded.shape) * 1e-2
    points.append(pt)
M.points = points


# run RNN on new points
rnn.run_points(n_seconds=600, cut=True)


viz = Visualizer(M, rnn=rnn, pca_sample_points=pca_sample_points)
viz.show(x_range=[0.1, 0.2], scale=0.0)
