import sys

sys.path.append("./")

from vedo import screenshot

import numpy as np

from manifold import Plane, Visualizer
from manifold._embeddings import Embedding
from manifold.decorators import parse2D
import manifold


manifold.visualize.reco_surface_radius = 0.05 * 3
manifold.visualize.point_size = 0.035
manifold.tangent_vector_radius = 0.001
"""
    Visualize two differently oriented manifolds with tangent vecors
"""


@parse2D
def plane_to_r3_flat(p0, p1):
    return (p0 + 1, p1 + 1, 0)


def vfield(point):
    return (
        np.sin(point[0] * np.pi * 2),
        np.sin(point[1] * np.pi * 2),
    )


plane_3d_embedding = Embedding("plane 3d", plane_to_r3_flat)
M = Plane(plane_3d_embedding, n_sample_points=16)
# M.points = [M.points[33]]
M.vectors_field = vfield

viz = Visualizer(M, manifold_alpha=1, wireframe=False)


cam = dict(
    pos=(1.5, 1.5, 8.01),
    focalPoint=(1.45, 1.35, -0.577),
    viewup=(0, -1.00, 0),
    distance=8.59,
    clippingRange=(7.22, 8.36),
)

viz.show(
    x_range=[0.1, 0.2],
    scale=0.1,
    show_basis_vecs=False,
    show_tangents=True,
    show_points=False,
    axes=0,
    camera=cam,
)


screenshot("paper/images/D.png")
