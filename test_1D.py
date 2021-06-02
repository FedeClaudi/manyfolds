from manifold import embeddings, Line, Circle, Visualizer
from manifold.rnn import RNN

# from manifold import vectors_fields

N = 3
K = 64

MANIFOLD = "circle"

if MANIFOLD == "line":
    M = Line(embeddings.line_to_r3, n_sample_points=3)
elif MANIFOLD == "helix":
    M = Line(embeddings.helix_to_r3, n_sample_points=3)
elif MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_r3_flat, n_sample_points=1)


# define vector field
# M.vectors_field = vectors_fields.double_sin

# create RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K)
rnn.run_points(n_seconds=10)

# visualize in embedding
viz = Visualizer(M, rnn)
viz.show(x_range=0.07, scale=0.25)
