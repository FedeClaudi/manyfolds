import sys
from numpy import pi, sin, cos

sys.path.append("./")
from vedo import screenshot

from manifold import embeddings, Plane
from manifold.manifolds.vectors_fields import (
    sphere_base,
    sphere_equator,
    sphere_poles,
    sphere_twirl,
)

from manifold.manifolds._embeddings import parse2D
from manifold.visualize import Visualizer
from manifold import visualize
from manifold.rnn import RNN


visualize.reco_surface_radius = 0.5
visualize.point_size = 0.03
visualize.tangent_vector_radius = 0.015
visualize.rnn_trace_alpha = 0.62


N = 64
K = 12


def vfield_one(point):
    p0, p1 = point.coordinates
    return (sin(pi * p0) * 0.3, 1 * 0.3)


def vfield_two(point):
    p0, p1 = point.coordinates
    p0, p1 = p0 - 0.5, p1 - 0.5

    return (sin(pi * p0) * 0.3, cos(pi * p1) * 0.3)


def vfield_three(point):
    p0, p1 = point.coordinates

    return (sin(2 * pi * p1) * 0.3, sin(2 * pi * p0) * 0.3)


vector_fields = (vfield_one, vfield_two, vfield_three)
V = 2

M = Plane(embeddings.plane_to_rn_flat, n_sample_points=[8, 12])
M.vectors_field = vector_fields[V]
# M.points = M.points[15:-15]


# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=15, cut=True)


viz = Visualizer(M, rnn=rnn, axes=0, manifold_alpha=1, pca_sample_points=100)


cam = dict(
    pos=(0.0809, -0.234, 8.56),
    focalPoint=(1.19e-7, 0, 3.98e-15),
    viewup=(-0.656, 0.754, 0.0268),
    distance=8.56,
    clippingRange=(8.01, 9.27),
)

viz.show(scale=0.3, show_points=True, cam=cam)

screenshot(f"./paper/images/3B_vfield_{V}.png")
