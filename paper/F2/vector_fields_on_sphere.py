import sys

sys.path.append("./")
from vedo import screenshot

from manifold import embeddings, Sphere
from manifold.manifolds.vectors_fields import (
    sphere_base,
    sphere_equator,
    sphere_poles,
    sphere_twirl,
)

from manifold.manifolds._embeddings import parse2D
from manifold.visualize import Visualizer
from manifold import visualize

visualize.reco_surface_radius = 0.1
visualize.point_size = 0.03
visualize.tangent_vector_radius = 0.015


@parse2D
def embedding_one(p0, p1):
    return (p0, p1, p1 * p0 + 0.5)


vector_fields = (sphere_base, sphere_equator, sphere_poles, sphere_twirl)
V = 3


M = Sphere(embeddings.sphere_to_rn, n_sample_points=[14, 20])
M.vectors_field = vector_fields[V]
M.points = M.points[13:-13]


viz = Visualizer(M, axes=0, manifold_alpha=1, pca_sample_points=100)

# for point in viz.manifold.points:
#     viz.visualize_basis_vectors_at_point(point, scale=0.05, r=0.01)

viz.plotter.camera.SetPosition([-0.312, 1.166, -6.808])
viz.plotter.camera.SetFocalPoint([0.001, 0.006, -0.0])
viz.plotter.camera.SetViewUp([-0.999, 0.008, 0.047])
viz.plotter.camera.SetDistance(6.913)
viz.plotter.camera.SetClippingRange([4.398, 10.094])


viz.show(scale=0.3, show_points=True)

screenshot(f"./paper/images/{M.name}_vecs_{V}.png")
