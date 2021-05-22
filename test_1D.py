import matplotlib.pyplot as plt
import numpy as np

from manifold import embeddings, Line, Circle
from manifold.rnn import RNN
from manifold.maths import angle_between, min_distance_from_point


MANIFOLD = "line"

if MANIFOLD == "line":
    M = Line(embeddings.line_to_r3, n_sample_points=3)
else:
    M = Circle(embeddings.circle_to_r3, n_sample_points=3)

# create base functions at each point in the manfiold
M.get_base_functions()

# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=64, scale=10)
rnn.run_points(n_seconds=25)

# visualize in embedding
ax = M.visualize_embedded()
M.visualize_base_functions_at_point(ax, x_range=0.05, scale=0.5)

# visualize charts
M.visualize_charts()

# visualize RNN dynamics
rnn.plot_traces(ax)

for point in M.points:
    p = point.embedded
    for i in range(2000):
        vec = rnn.step(p)
        p = vec
    print(
        f"Vectors angle: {np.degrees(angle_between(vec, point.base_functions[0].tangent_vector)):.3f}"
    )
    print(
        f"Distance from manifold: {min_distance_from_point(M.embedded, vec):.3f}"
    )

plt.show()
