import sys

sys.path.append("./")

import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

from fcutils.plot.figure import clean_axes
from fcutils.plot.elements import plot_mean_and_error

from manifold import embeddings, Torus, Sphere, Cylinder
from manifold.rnn import RNN
from manifold import vectors_fields


plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update(
    {
        "font.size": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)


def register_in_time(trials, n_samples):
    """
        Given a list of 1d numpy arrays of different length,
        this function returns an array of shape (n_samples, n_traces)
    """
    target = np.zeros((n_samples, len(trials)))
    for trial_n, trial in enumerate(trials):
        n = len(trial)
        for i in range(n_samples):
            idx = int(np.floor(n * (i / n_samples)))
            target[i, trial_n] = trial[idx]
    return target


# --------------------------------- settings --------------------------------- #
MANIFOLDS = ("cylinder", "torus", "sphere")
N = 64
K = 12
n_sec = 150  # max n sec, each trace is only used until it goes back to the starting point
N_repeats = 10

colors = "rgb"
f, ax = plt.subplots(figsize=(16, 9))
for manifold, color in zip(MANIFOLDS, colors):
    logger.info(f"Running manifold {manifold}")
    if manifold == "sphere":
        M = Sphere(embeddings.sphere_to_rn, n_sample_points=5)
        M.vectors_field = vectors_fields.second_only

    elif manifold == "cylinder":
        M = Cylinder(embeddings.cylinder_to_rn, n_sample_points=5)
        M.vectors_field = vectors_fields.first_only
    else:
        M = Torus(embeddings.torus_to_rn, n_sample_points=5)
        M.vectors_field = vectors_fields.second_only

    origin_distances = []  # store distance from origin for each RNN trace
    for repeat in range(N_repeats):
        rnn = RNN(M, n_units=N)
        rnn.traces = []
        rnn.build_W(k=K)
        rnn.run_points(n_seconds=n_sec, cut=False)

        # cut each trace once it completed a whole radius and get distance from origin
        for trace in rnn.traces:
            dist_from_start = np.apply_along_axis(
                np.linalg.norm, 1, trace.trace - trace.initial_condition
            )
            back_at_start = np.argmin(dist_from_start[500:]) + 500

            # to check that we are cutting the traces correctly
            # ax.plot(dist_from_start, lw=.5, color='k')
            # ax.plot(dist_from_start[:back_at_start], lw=1, color='r')

            dist = np.apply_along_axis(
                np.linalg.norm, 1, trace.trace[:back_at_start]
            )
            origin_distances.append(dist / dist[0])

    # plot
    distances = register_in_time(origin_distances, 360)
    plot_mean_and_error(
        distances.mean(1),
        np.std(distances, 1),
        ax,
        label=manifold,
        color=color,
    )

ax.legend()
ax.axhline(1, lw=1, ls=":", color="k")
ax.set(
    xlabel="position",
    xticks=[0, 360],
    ylim=[0.8, 1.2],
    xticklabels=["0", "$\\pi$"],
    ylabel="normalized distance",
)
plt.show()
clean_axes(f)
f.savefig(f"paper/images/3D_drift.svg", format="svg")
