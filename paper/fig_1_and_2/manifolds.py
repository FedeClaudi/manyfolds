import sys
import numpy as np

sys.path.append("./")

from functools import partial
from vedo import screenshot

from manifold import (
    embeddings,
    Line,
    Circle,
    Visualizer,
    Plane,
    Torus,
    Sphere,
    Cylinder,
)
from manifold.maths import ortho_normal_matrix
from manifold import visualize

# visualize.reco_surface_radius = 0.02


def embedding_one(mtx, p):
    e1 = (p[0], p[1], p[1] * p[0] + 0.5)
    return tuple(mtx @ np.array(e1))


def prepare_plane_to_rn(n=64):
    mtx = ortho_normal_matrix(n, 3)
    return partial(embedding_one, mtx)


# from manifold import vectors_fields
MANIFOLDS = (
    "line",
    "helix",
    "circle",
    "curvy circle",
    "plane",
    "plane flat",
    "torus",
    "thin torus",
    "sphere",
    "ellipse",
    "cylinder",
    "cone",
)
manifold_number = 9
n_dims = 64


MANIFOLD = MANIFOLDS[manifold_number]
if MANIFOLD == "line":
    M = Line(embeddings.line_to_rn, n_sample_points=3)

elif MANIFOLD == "helix":
    M = Line(embeddings.helix_to_rn, n_sample_points=3)

elif MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_rn_bent, n_sample_points=4)

elif MANIFOLD == "curvy circle":
    M = Circle(embeddings.circle_to_rn, n_sample_points=4)

elif MANIFOLD == "plane":
    M = Plane(embeddings.plane_to_rn, n_sample_points=[3, 2])
    M.points = [M.points[7]]

elif MANIFOLD == "plane flat":
    M = Plane(embeddings.plane_to_rn_flat, n_sample_points=[3, 2])
    M.points = [M.points[7]]

elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_rn, n_sample_points=[5, 14])
    M.points = [M.points[7]]
elif MANIFOLD == "thin torus":
    M = Torus(embeddings.thin_torus_to_rn, n_sample_points=[5, 14])
    M.points = [M.points[7]]

elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_rn, n_sample_points=[4, 10])
    M.points = [M.points[7]]

elif MANIFOLD == "ellipse":
    M = Sphere(embeddings.ellipse_to_rn, n_sample_points=[4, 10])
    M.points = [M.points[7]]

elif MANIFOLD == "cylinder":
    M = Cylinder(embeddings.cylinder_to_rn, n_sample_points=[6, 2])
    M.points = [M.points[7]]

elif MANIFOLD == "cone":
    M = Cylinder(embeddings.cone_to_rn, n_sample_points=[6, 2],)
    M.points = [M.points[7]]


if M.d == 1:
    visualize.point_size = 0.05
    visualize.manifold_1d_r = 0.03
    visualize.tangent_vector_radius = 0.05  # 0.04
else:
    # visualize.point_size = 0.1  # 0.05
    visualize.manifold_1d_r = 0.05  # 0.03
    visualize.tangent_vector_radius = 0.05  # 0.04
    visualize.reco_surface_radius = 0.2


viz = Visualizer(
    M, axes=0, wireframe=False, pca_sample_points=100, manifold_alpha=1
)

viz.visualize_manifold()

for point in viz.manifold.points:
    viz.visualize_basis_vectors_at_point(point, scale=0.4)

for actor in viz.actors:
    actor.lighting("off")

if M.d == 2:
    viz._add_silhouette(viz.manifold_actor, lw=3)

viz.plotter.camera.SetPosition([-0.741, -7.359, 6.417])
viz.plotter.camera.SetFocalPoint([-0.005, -0.094, 0.525])
viz.plotter.camera.SetViewUp([0.018, 0.629, 0.777])
viz.plotter.camera.SetDistance(9.383)
viz.plotter.camera.SetClippingRange([6.5, 13.036])

viz.plotter.show(*viz.actors)

screenshot(f"./paper/fig_1_and_2/{M.name}_{M.embedding.name}.png")

viz.plotter.close()
