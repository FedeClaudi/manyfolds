import sys

sys.path.append("./")

from vedo import screenshot

from manifold import embeddings, Circle, Visualizer, Torus, Sphere
from manifold.rnn import RNN
from manifold import vectors_fields


# from manifold import vectors_fields

MANIFOLD = "sphere"
K = 32

if MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_r3_angled, n_sample_points=0)
elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[8, 0])
    x_range = [0.1, 0.05]
    M.vectors_field = vectors_fields.second_only

elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[10, 0])
    x_range = [0.02, 0.02]
    M.vectors_field = vectors_fields.second_only


# create RNN
rnn = RNN(M, n_units=3)
rnn.build_W(k=K, scale=100)
rnn.run_points(n_seconds=10)

# visualize in embedding
viz = Visualizer(
    M,
    rnn,
    axes=0,
    manifold_color="#b8b6d1",
    point_color="#3838BA",
    wireframe=False,
    manifold_alpha=0.5,
)
viz.show(x_range=0.07, scale=0.25)


screenshot(f"./paper/figure_3/{M.name}_flows.png")
