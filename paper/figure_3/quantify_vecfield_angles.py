import sys

sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np

from fcutils.plot.figure import clean_axes

from manifold import Plane, embeddings, vectors_fields
from manifold.rnn import RNN
from manifold.tangent_vector import get_tangent_vector
from manifold.maths import angle_between


plt.rc("text", usetex=True)
plt.rc("font", family="serif")

N = 12
K = 32
n_points = 40
n_sec = 0.25


f, ax = plt.subplots()
clean_axes(f)


fields = dict(
    identity=vectors_fields.identity,
    first=vectors_fields.first_only,
    second=vectors_fields.second_only,
)

bins = np.linspace(0, 1, 20)
for F in fields.keys():
    M = Plane(embeddings.prepare_plane_to_rn(n=N), n_sample_points=n_points)
    M.vectors_field = fields[F]

    # create RNN
    rnn = RNN(M, n_units=3)
    rnn.traces = []
    rnn.build_W(k=K, scale=100)
    rnn.run_points(n_seconds=n_sec)

    # get angles
    angles = []
    for n, trace in enumerate(rnn.traces):
        rnn_vec = trace.trace[-1] - trace.trace[0]
        if np.linalg.norm(rnn_vec) == 0:
            continue
        tan_vec = get_tangent_vector(M.points[n], vectors_field=fields[F])

        angles.append(angle_between(rnn_vec, tan_vec))

    ax.hist(angles, label=f"{F}", bins=bins, alpha=0.5, density=True)
ax.legend()
plt.show()
ax.set(xlabel="$\theta (rad)$", ylabel="density")
f.savefig(f"paper/figure_3/angles_N_{N}.svg", format="svg")


# viz = Visualizer(M, rnn, axes=0, manifold_color='#b8b6d1', point_color='#3838BA', wireframe=False, manifold_alpha=.5)
# for point in viz.manifold.points:
#     viz.visualize_basis_vectors_at_point(point, scale=.15, r=0.015)


# viz.plotter.camera.SetPosition( [-2.728, -2.351, 1.512] )
# viz.plotter.camera.SetFocalPoint( [-0.096, 0.009, 0.247] )
# viz.plotter.camera.SetViewUp( [0.292, 0.178, 0.94] )
# viz.plotter.camera.SetDistance( 3.755 )
# viz.plotter.camera.SetClippingRange( [1.795, 6.231] )

# viz.show(scale=.2)
