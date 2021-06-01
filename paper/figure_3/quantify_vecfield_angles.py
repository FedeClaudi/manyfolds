import sys

sys.path.append("./")

import matplotlib.pyplot as plt

from manifold import Plane, embeddings, vectors_fields
from manifold.rnn import RNN
from manifold.tangent_vector import get_tangent_vector
from manifold.maths import angle_between


N = 12
K = 32
n_points = 20
F = "identity"
f, ax = plt.subplots()


fields = dict(
    identity=vectors_fields.identity,
    first=vectors_fields.first_only,
    second=vectors_fields.second_only,
)

for n_sec in (0.15, 0.5, 1):
    for F in fields.keys():
        M = Plane(
            embeddings.prepare_plane_to_rn(n=N), n_sample_points=n_points
        )
        M.vectors_field = fields[F]

        # create RNN
        rnn = RNN(M, n_units=3)
        rnn.traces = []
        rnn.build_W(k=K, scale=100)
        rnn.run_points(n_seconds=0.15)

        # get angles
        angles = []
        for n, trace in enumerate(rnn.traces):
            rnn_vec = trace.trace[-1] - trace.trace[0]
            tan_vec = get_tangent_vector(M.points[n], vectors_field=fields[F])

            angles.append(angle_between(rnn_vec, tan_vec))

        ax.hist(angles, label=f"{F}_nsec: {n_sec}")
ax.legend()
plt.show()

# viz = Visualizer(M, rnn, axes=0, manifold_color='#b8b6d1', point_color='#3838BA', wireframe=False, manifold_alpha=.5)
# for point in viz.manifold.points:
#     viz.visualize_basis_vectors_at_point(point, scale=.15, r=0.015)


# viz.plotter.camera.SetPosition( [-2.728, -2.351, 1.512] )
# viz.plotter.camera.SetFocalPoint( [-0.096, 0.009, 0.247] )
# viz.plotter.camera.SetViewUp( [0.292, 0.178, 0.94] )
# viz.plotter.camera.SetDistance( 3.755 )
# viz.plotter.camera.SetClippingRange( [1.795, 6.231] )

# viz.show(scale=.2)
