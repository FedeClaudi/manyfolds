import sys

sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from random import choices
from loguru import logger

from fcutils.progress import track
from fcutils.plot.elements import plot_mean_and_error
from fcutils.plot.figure import clean_axes

from manifold import embeddings, Circle, Torus, Sphere
from manifold.rnn import RNN
from manifold import vectors_fields

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

N = 64
K = 12
n_sec = 60
NT = 32  # number of sample points per run
nreps = 30

f, ax = plt.subplots(figsize=(16, 9))
clean_axes(f)


# from manifold import vectors_fields
MANIFOLDS = ("circle", "torus", "sphere")
colors = "rgb"
n_frames = int(n_sec * (1 / RNN.dt))

for n, MANIFOLD in enumerate(MANIFOLDS):
    logger.info(MANIFOLD)
    if MANIFOLD == "circle":
        if N == 3:
            M = Circle(embeddings.circle_to_r3_flat, n_sample_points=NT)
        else:
            M = Circle(
                embeddings.prepare_circle_angled_to_rn(n=N), n_sample_points=NT
            )

    elif MANIFOLD == "torus":
        if N == 3:
            M = Torus(embeddings.torus_to_r3, n_sample_points=[NT, 0])
        else:
            M = Torus(
                embeddings.prepare_torus_to_rn(n=N), n_sample_points=[NT, 0]
            )
        M.vectors_field = vectors_fields.second_only

    elif MANIFOLD == "sphere":
        if N == 3:
            M = Sphere(embeddings.sphere_to_r3, n_sample_points=[NT, 0])
        else:
            M = Sphere(
                embeddings.prepare_sphere_to_rn(n=N), n_sample_points=[NT, 0]
            )
        M.vectors_field = vectors_fields.second_only

    # create RNN
    all_distances = np.zeros((nreps, n_frames, NT))
    for rep in range(nreps):
        logger.info(f"Rep: {rep}")
        rnn = RNN(M, n_units=N)
        rnn.build_W(k=K)
        rnn.run_points(n_seconds=n_sec, cut=False)

        # compute distance from manifold
        traces = choices(rnn.traces, k=NT)
        distances = np.zeros((n_frames, NT))

        for i, trace in track(
            enumerate(traces),
            total=NT,
            description=f"Computing distances for {M.name}",
        ):
            n0 = np.linalg.norm(trace.trace[0, :])
            distances[:, i] = (
                np.apply_along_axis(np.linalg.norm, 1, trace.trace) / n0
            )
        all_distances[rep, :, :] = distances

    # ----------------------------------- plot ----------------------------------- #
    distances = np.mean(all_distances, 0)
    plot_mean_and_error(
        np.mean(distances, 1),
        np.std(distances, 1),
        ax,
        label=f"${M.name}$",
        color=colors[n],
        lw=1,
    )

ax.axhline(1, lw=3, ls="--", color=[0.6, 0.6, 0.6], zorder=-1)
ax.set(
    ylabel="distance (norm)",
    xlabel="time (s)",
    ylim=[0.8, 1.25],
    xticks=np.arange(0, n_sec + 1, 10) * int(1 / rnn.dt),
    xticklabels=np.arange(0, n_sec + 1, 10),
)
ax.legend()
plt.show()

f.savefig(f"paper/figure_3/drift_N_{N}.svg", format="svg")
