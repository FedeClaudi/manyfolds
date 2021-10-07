import sys

sys.path.append("./")

from vedo import screenshot

from manifold import Plane, Visualizer
from manifold._embeddings import Embedding
from manifold.decorators import parse2D
import manifold

manifold.visualize.reco_surface_radius = 0.025
manifold.visualize.point_size = 0.035

"""
    Visualize two differently oriented manifolds with tangent vecors
"""


@parse2D
def plane_to_r3_flat(p0, p1):
    return (p0 + 1, p1 + 1, 0)


plane_3d_embedding = Embedding("plane 3d", plane_to_r3_flat)
M = Plane(plane_3d_embedding, n_sample_points=8)
M.points = [M.points[33]]


viz = Visualizer(M, manifold_alpha=1, wireframe=False)
viz.show(
    x_range=[0.1, 0.2],
    scale=0.2,
    show_basis_vecs=True,
    show_tangents=False,
    show_points=False,
    axes=0,
)


screenshot("paper/images/B.png")
