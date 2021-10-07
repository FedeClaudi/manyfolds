import sys
import numpy as np

sys.path.append("./")
from vedo import screenshot
from vedo.shapes import Tube

from myterial import salmon

from manifold import embeddings, Plane
from manifold.visualize import Visualizer
from manifold import visualize
from manifold.rnn import RNN


"""
    3D viisualization of an RNN's dynamics over time fitted
    to the plane with a single fixed point attractor
    at the center.
"""

visualize.reco_surface_radius = 0.5
visualize.point_size = 0.03
visualize.tangent_vector_radius = 0.015
visualize.rnn_trace_alpha = 0.62


N = 64
K = 12


def vfield(point):
    # fixed point at center
    p0, p1 = point.coordinates
    return ((0.5 - p0) * 3, (0.5 - p1) * 3)


M = Plane(embeddings.plane_to_rn_flat, n_sample_points=12)
M.vectors_field = vfield


# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=60, cut=False)


# visualize vector field
viz = Visualizer(M, rnn=None, axes=0, manifold_alpha=1, pca_sample_points=100)
cam = dict(
    pos=(-7.84, -8.65, 3.76),
    focalPoint=(2.38e-7, 0, 1.49),
    viewup=(0.0954, 0.171, 0.981),
    distance=11.9,
    clippingRange=(6.02, 19.3),
)

for trace in rnn.traces:
    trace_pc = viz.pca.transform(trace.trace)
    coords = trace_pc.copy()
    coords[:, 2] = np.linspace(0, 3, len(coords))
    tube = Tube(coords, c=salmon, r=0.01, alpha=1)
    viz.actors.append(tube)
    viz._add_silhouette(tube)

# show vector field
viz.show(scale=0.15, show_points=True, camera=cam)
screenshot(f"./paper/images/3D_vfield_.png")
