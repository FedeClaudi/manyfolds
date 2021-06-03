import sys

sys.path.append("./")
from vedo import settings, screenshot

from manifold import embeddings, Sphere
from manifold.manifolds.vectors_fields import (
    sphere_base,
    sin_on_sphere,
    sphere_base2,
)
from manifold.tangent_vector import get_basis_tangent_vector
from manifold.manifolds.embeddings import parse2D
from manifold.visualize import Visualizer


@parse2D
def embedding_one(p0, p1):
    return (p0, p1, p1 * p0 + 0.5)


vector_fields = (sphere_base, sin_on_sphere, sphere_base2)
V = 0


M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 12])
M.vectors_field = vector_fields[V]


viz = Visualizer(
    M, axes=0, wireframe=False, manifold_color="#b8b6d1", point_color="#3838BA"
)

viz.visualize_manifold()
viz.visualize_tangent_vectors(scale=0.15, x_range=0.1)

for point in viz.manifold.points:
    pt = point.embedded
    for fn in point.base_functions:
        fn.embedd()
        vec = get_basis_tangent_vector(point, fn) * 0.1
        viz._render_cylinder([pt, pt + vec], "#000000", r=0.015, alpha=1)

for actor in viz.actors:
    actor.lighting("off")

if M.d == 2:
    viz._add_silhouette(viz.manifold_actor, lw=3)

viz.plotter.camera.SetPosition([-0.882, -5.163, 0.246])
viz.plotter.camera.SetFocalPoint([0.0, 0.0, 0.0])
viz.plotter.camera.SetViewUp([0.032, 0.042, 0.999])
viz.plotter.camera.SetDistance(5.244)
viz.plotter.camera.SetClippingRange([3.286, 7.721])

viz.plotter.show(*viz.actors)

settings.screenshotTransparentBackground = 1
screenshot(f"./paper/fig_1_and_2/{M.name}_vecs_{V}.png")

viz.plotter.close()
