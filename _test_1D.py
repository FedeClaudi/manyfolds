import matplotlib.pyplot as plt
import numpy as np

from manifold.manifolds import Line, Circle
from manifold.rnn import RNN
from manifold.maths import angle_between, min_distance_from_point

from numpy import sin, cos

MANIFOLD = "line"

if MANIFOLD == "line":
    # ? curved line embedding
    def embedding(p):
        if not isinstance(p, float):
            p = p[0]
        return (sin(2 * p), sin(p), -cos(p))

    # ? straight line embedding
    # def embedding(p):
    #     if not isinstance(p, float):
    #         p = p[0]
    #     return (p, -p, 2 * p)

    M = Line(embedding, n_sample_points=3)
else:

    def embedding(p):
        if not isinstance(p, float):
            p = p[0]
        return (sin(p) / 2, cos(p) / 2, 1)

    M = Circle(embedding, n_sample_points=3)

# create base functions at each point in the manfiold
M.get_base_functions()


# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=40, scale=10)
rnn.run_points(n_seconds=10)

# visualize in embedding
ax = M.visualize_embedded()
M.visualize_base_functions_at_point(ax, x_range=0.05)

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
