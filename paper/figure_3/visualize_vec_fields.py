import sys

sys.path.append("./")

from vedo import screenshot

from manifold import embeddings, Circle, Visualizer, Torus, Sphere
from manifold.rnn import RNN
from manifold import vectors_fields
from manifold import visualize

visualize.reco_surface_radius = 0.2
visualize.rnn_trace_radius = 0.035
visualize.tangent_vector_radius = 0.04

# from manifold import vectors_fields

MANIFOLD = "sphere"
F = "second_only"

N = 64
K = 15

fields = dict(
    sin=vectors_fields.sin,
    double_sin=vectors_fields.double_sin,
    first_only=vectors_fields.torus_first,
    second_only=vectors_fields.second_only,
    torus_second=vectors_fields.torus_second,
    sphere_poles=vectors_fields.sphere_poles,
    sphere_equator=vectors_fields.sphere_equator,
    sphere_base=vectors_fields.sphere_base,
    sphere_base_2=vectors_fields.sphere_base2,
)


if MANIFOLD == "circle":
    M = Circle(embeddings.prepare_circle_angled_to_rn(n=N), n_sample_points=18)
    cam = None

elif MANIFOLD == "torus":
    M = Torus(embeddings.prepare_torus_to_rn(n=N), n_sample_points=[4, 2])

    cam = dict(
        pos=(-0.123, -3.90, 2.70),
        focalPoint=(7.67e-3, -6.15e-4, 1.93e-6),
        viewup=(0.0140, 0.569, 0.822),
        distance=4.75,
        clippingRange=(2.87, 7.12),
    )


elif MANIFOLD == "sphere":
    # sphere_equator, sphere_base and torus_second
    M = Sphere(embeddings.prepare_sphere_to_rn(n=N), n_sample_points=[12, 0])
    cam = dict(
        pos=(2.34, -12.8, 3.81),
        focalPoint=(6.89e-3, 0.0454, -5.48e-3),
        viewup=(0.983, 0.145, -0.114),
        distance=13.6,
        clippingRange=(7.75, 21.0),
    )


M.vectors_field = fields[F]


# create RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=5)

# visualize in embedding
viz = Visualizer(M, rnn, camera=cam, axes=0, manifold_alpha=0.5,)

for point in M.points:
    viz.visualize_basis_vectors_at_point(
        point, r=0.02, scale=0.2, color=[0.2, 0.2, 0.2]
    )

viz.show(x_range=0.07, scale=0.3)


screenshot(f"./paper/figure_3/{M.name}_fields_{F}.png")
