import sys

sys.path.append("./")


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

from manifold.manifolds.embeddings import parse2D


@parse2D
def embedding_one(p0, p1):
    return (p0, p1, p1 * p0 + 0.5)


# from manifold import vectors_fields
MANIFOLDS = ("line", "circle", "plane", "torus", "sphere", "cylinder")
N = 0

MANIFOLD = MANIFOLDS[N]
if MANIFOLD == "line":
    M = Line(embeddings.line_to_r3, n_sample_points=3)
    M.points = [M.points[2]]

elif MANIFOLD == "circle":
    M = Circle(embeddings.circle_to_r3, n_sample_points=4)

elif MANIFOLD == "plane":
    M = Plane(embedding_one, n_sample_points=[3, 2])
    M.points = [M.points[4]]

elif MANIFOLD == "torus":
    M = Torus(embeddings.torus_to_r3, n_sample_points=[8, 8])
    M.points = [M.points[15]]


elif MANIFOLD == "sphere":
    M = Sphere(embeddings.sphere_to_r3, n_sample_points=[4, 10])
    M.points = [M.points[29]]

elif MANIFOLD == "cylinder":
    M = Cylinder(embeddings.cylinder_to_r3, n_sample_points=[6, 2])
    M.points = [M.points[7]]


viz = Visualizer(
    M, axes=0, wireframe=False, manifold_color="#b8b6d1", point_color="#3838BA"
)

viz.visualize_manifold()

for point in viz.manifold.points:
    viz.visualize_basis_vectors_at_point(point)


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

screenshot(f"./paper/fig_1_and_2/{M.name}.png")

viz.plotter.close()
