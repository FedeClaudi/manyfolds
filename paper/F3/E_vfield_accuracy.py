import sys

sys.path.append("./")
from numpy import pi, sin, cos

from fcutils.plot.figure import clean_axes
from fcutils.plot.distributions import plot_kde
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


from manifold import embeddings, Plane
from manifold.rnn import RNN
from manifold.tangent_vector import get_tangent_vector
from manifold.maths import angle_between


N = 64
K = 12
n_sec = 5
N_repeats = 10


def vfield_one(point):
    p0, p1 = point.coordinates
    return (sin(pi * p0) * 0.3, 1 * 0.3)


def vfield_two(point):
    p0, p1 = point.coordinates
    p0, p1 = p0 - 0.5, p1 - 0.5

    return (sin(pi * p0) * 0.3, cos(pi * p1) * 0.3)


def vfield_three(point):
    p0, p1 = point.coordinates

    return (sin(2 * pi * p1) * 0.3, sin(2 * pi * p0) * 0.3)


vector_fields = dict(F1=vfield_one, F2=vfield_two, F3=vfield_three)

f, ax = plt.subplots(figsize=(9, 9))
clean_axes(f)

bins = np.linspace(0, 1, 20)
for F in vector_fields.keys():
    logger.info(f"Running vfield: {F}")
    angles = []
    for i in range(N_repeats):
        M = Plane(embeddings.plane_to_rn_flat, n_sample_points=[10, 10])
        M.vectors_field = vector_fields[F]

        # fit and run RNN
        rnn = RNN(M, n_units=N)
        rnn.traces = []
        rnn.build_W(k=K, scale=1)
        rnn.run_points(n_seconds=n_sec, cut=True)

        for n, trace in enumerate(rnn.traces):
            rnn_vec = trace.trace[-1] - trace.trace[0]
            tan_vec = get_tangent_vector(
                M.points[n], vectors_field=vector_fields[F]
            )

            if (
                np.linalg.norm(rnn_vec) <= 1e-4
                or np.linalg.norm(tan_vec) <= 1e-4
            ):
                logger.debug("skipping because vectors too short")
                continue

            angles.append(angle_between(rnn_vec, tan_vec))

    plot_kde(ax=ax, data=angles, label=f"{F}", kde_kwargs=dict(bw=0.02))
ax.legend()
ax.set(xlabel=r"$\theta (rad)$", ylabel="density")
f.savefig(f"paper/images/3E_vfield_accuracy.svg", format="svg")
plt.show()
