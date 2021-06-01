import sys
import numpy as np

sys.path.append("./")

from manifold import Plane
from manifold.tangent_vector import get_basis_tangent_vector
from manifold import Visualizer
from manifold.manifolds.embeddings import parse2D


@parse2D
def embedding_one(p0, p1):
    return (p0, p1, p1 * p0 + 0.5)


M = Plane(embedding_one, n_sample_points=[5, 5])
M.points = [M.points[14]]
x_range = [0.3, 0.3]


# visualize in embedding
viz = Visualizer(
    M, axes=1, wireframe=False, manifold_color="#D5D3E9", point_color="#3838BA"
)

viz.visualize_manifold()

for point in viz.manifold.points:
    pt = np.array(point.embedded)
    for fn in point.base_functions:
        fn.embedd()
        vec = get_basis_tangent_vector(point, fn) * 0.25
        viz._render_cylinder([pt, pt + vec], "#000000")
viz.actors.append(viz.manifold_actor.addShadow(z=0, alpha=0.5))

for actor in viz.actors:
    actor.lighting("off")

viz.plotter.camera.SetPosition([-4.046, 0.936, 1.728])
viz.plotter.camera.SetFocalPoint([0.497, 0.45, 0.799])
viz.plotter.camera.SetViewUp([0.202, 0.016, 0.979])
viz.plotter.camera.SetDistance(4.663)
viz.plotter.camera.SetClippingRange([2.976, 6.798])
viz.plotter.show(*viz.actors)
