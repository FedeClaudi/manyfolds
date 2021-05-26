from manifold import embeddings, Line, Circle
from manifold.rnn import RNN

# from manifold import vectors_fields

MANIFOLD = "helix"

if MANIFOLD == "line":
    M = Line(embeddings.line_to_r3, n_sample_points=3)
elif MANIFOLD == "helix":
    M = Line(embeddings.helix_to_r3, n_sample_points=3)

elif MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_r3_angled, n_sample_points=8)
M.print_embedding_bounds()

# define vector field
# M.vectors_field = vectors_fields.sin


# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=24, scale=0.01)
rnn.run_points(n_seconds=2)

# visualize in embedding
M.visualize_embedded()
M.visualize_base_functions_at_point(x_range=0.07, scale=0.25)

# visualize charts
# M.visualize_charts()

# visualize RNN dynamics
rnn.plot_traces()

M.show()
