import autograd.numpy as np

from autograd import jacobian

from manifold import embeddings, Circle
from manifold.rnn import RNN

# from manifold import vectors_fields
from manifold import Visualizer
from manifold.maths import unit_vector

# --------------------------------- settings --------------------------------- #
N = 3
K = 6

M = Circle(embeddings.circle_to_r3_angled, n_sample_points=0)
# M.vectors_field = vectors_fields.sin
# M = Sphere(embeddings.sphere_to_r3, n_sample_points=[2, 0])
# M.vectors_fields = vectors_fields.second_only
pca_sample_points = 75

M.print_embedding_bounds()

# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=90, cut=True)

viz = Visualizer(M, rnn=rnn, pca_sample_points=pca_sample_points)

# improve W's Jacobian SVD
# for point in M.points:
colors = "rgby"
for t, trace in enumerate(rnn.traces):
    if t == 0:
        continue
    for n in range(len(trace)):
        if n % 100 == 0:
            point = trace.trace[n]
            # compute jacobian
            J = jacobian(rnn.dynamics)(point)

            # do SVD of jacobian
            u, s, _ = np.linalg.svd(J)
            evals, evecs = np.linalg.eig(J)
            evals = np.real(evals)
            evecs = np.real(evecs)

            idx = evals.argsort()[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            # draw eigenvectors
            # pt_lowd = viz.pca.transform(point.reshape(1, -1))[0]
            # tangent = get_tangent_vector(point, M.vectors_field)
            # for i in range(N):
            #     vec = np.real(evecs[:, i])

            # vec = u[:, i] * s[i]  * 1e-3
            # vec_lowd = viz.pca.transform((vec + point).reshape(1, -1))[0]

            # if evals[i] == np.max(evals):
            #     color='m'
            #     r = 0.01
            #     vec = vec
            # else:
            # continue
            # color, r = 'k', 0.03
            # viz._render_cylinder([point, vec + point], color=color, r=r, alpha=.5)
            # if i == 3:
            #     break
            viz._render_cylinder(
                [point, unit_vector(evals.dot(evecs.T)) + point],
                color="b",
                r=0.015,
            )

    break

    a = 1


viz.show(x_range=[0.1, 0.2], scale=0.2)
