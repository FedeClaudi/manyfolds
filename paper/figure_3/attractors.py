import sys

sys.path.append("./")

from manifold import embeddings, Circle
from manifold.rnn import RNN

from manifold import vectors_fields
from manifold import Visualizer
from manifold import visualize

visualize.rnn_trace_radius = 0.02

N = 64
K = 14

M = Circle(embeddings.prepare_circle_angled_to_rn(n=N), n_sample_points=8)
pca_sample_points = 50
M.vectors_field = vectors_fields.cos


M.print_embedding_bounds()

# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=10, cut=True)


viz = Visualizer(M, rnn=rnn, pca_sample_points=pca_sample_points)
viz.show(x_range=[0.1, 0.2], scale=0.2)
