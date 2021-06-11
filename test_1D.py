from manifold import embeddings, Line, Circle, Visualizer

# from manifold.rnn import RNN
from numpy import cos, sin
import numpy as np

# from vedo import Tube

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


# from sympy import diff
# from sympy.abc import p
# from sympy import sin, cos


def pushforward(p):
    return (cos(p), -sin(p), cos(p))


# def emb(p):
#     return (sin(p), cos(p))

# print(diff(pushforward(p)))

# define vector field
# M.vectors_field = vectors_fields.double_sin

# create RNN
# rnn = RNN(M, n_units=N)
# rnn.build_W(k=K)
# rnn.run_points(n_seconds=0.5)

# visualize in embedding
viz = Visualizer(M, None)


for point in M.points:
    coords = point.base_functions[0].get_manifold_coordinates()
    fn = (
        np.apply_along_axis(pushforward, 1, coords)
        .astype(np.float64)[50, :]
        .ravel()
    )

    # viz.actors.append(Tube(fn, r=0.02))
    viz._render_cylinder([point.embedded, point.embedded + fn], "red")

viz.show(x_range=0.07, scale=0.25)
