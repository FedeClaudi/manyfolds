from manifold import embeddings, Line, Circle, Visualizer
from manifold.rnn import RNN

# from manifold import vectors_fields

MANIFOLD = "circle"

if MANIFOLD == "line":
    M = Line(embeddings.line_to_r3, n_sample_points=3)
elif MANIFOLD == "helix":
    M = Line(embeddings.helix_to_r3, n_sample_points=3)
elif MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_r3_flat, n_sample_points=8)


# define vector field
# M.vectors_field = vectors_fields.sin


# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=64, scale=100)
rnn.run_points(n_seconds=10)

# visualize in embedding
viz = Visualizer(M, rnn)
viz.show(x_range=0.07, scale=0.75)
