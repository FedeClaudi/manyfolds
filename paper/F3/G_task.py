import sys

sys.path.append("./")
import numpy as np

import matplotlib.pyplot as plt
from fcutils.plot.figure import clean_axes

# from vedo import screenshot
from myterial import amber_darker, pink_darker, salmon_darker

from manifold import Plane
from manifold.rnn import RNN

# from manifold.manifolds.embeddings import plane_to_rn_flat
from manifold import Visualizer
from manifold import visualize

visualize.rnn_trace_alpha = 0.8
visualize.reco_surface_radius = 0.0


N = 64
K = 12

trial_n_sec = 120
B_scale = 5


cam = dict(
    pos=(-0.325, -0.171, 5.77),
    focalPoint=(0.0535, -0.0282, 2.81e-3),
    viewup=(0.317, 0.947, 0.0443),
    distance=5.78,
    clippingRange=(5.38, 6.29),
)

# ------------------------------ manifolds code ------------------------------ #


def vfield(p):
    s1 = np.sin(2.5 * np.pi * (p[0] - 0.1)) * 0.6
    s2 = np.sin(2.5 * np.pi * (p[1] - 0.1)) * 0.6
    return (-s1, -s2)


class Embedding:
    N = 64
    name = "plane"

    def __init__(self):
        # select a random vector
        v1 = np.random.rand(N)
        v1 /= np.linalg.norm(v1)

        # get an orthogonal vector
        v2 = np.random.randn(N)  # take a random vector
        v2 -= v2.dot(v1) * v1  # make it orthogonal to k
        v2 /= np.linalg.norm(v2)  # normalize it

        self.v1 = v1
        self.v2 = v2

    def __call__(self, p):
        return (
            np.array((self.v1 * (p[0] + 0.2) + self.v2 * (p[1] + 0.2))) - 0.4
        )


# --------------------------------- task code -------------------------------- #
def generate_trial():
    n_secs = trial_n_sec
    n_steps = int(n_secs / RNN.dt)

    inputs = np.zeros((n_steps, 2))
    outputs = np.zeros_like(inputs)

    for i in range(2):
        # ever n sec flip the inputs
        f = 0.17
        flips = np.random.uniform(1, n_steps - 1, int(f / RNN.dt)).astype(
            np.int32
        )
        flips2 = np.random.uniform(1, n_steps - 1, int(f / RNN.dt)).astype(
            np.int32
        )

        for shift in (0, 1, 2, 3):
            inputs[flips + shift, i] = 1
            inputs[flips2 + shift, i] = -1

        # store the state
        state = 0
        for n, x in enumerate(inputs[:, i]):
            if x == 1:
                state = 1
            elif x == -1:
                state = -1
            outputs[n, i] = state

    return inputs, outputs


# ------------------------------ create manifold ----------------------------- #
phi = Embedding()
M = Plane(phi, n_sample_points=[8, 8])
M.print_embedding_bounds()
M.vectors_field = vfield
# ---------------------------------- fir RNN --------------------------------- #
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=50, cut=False)

rnn.B = np.vstack([phi.v1, phi.v2]).T * B_scale

# --------------------------------- visualize -------------------------------- #
viz = Visualizer(
    M,
    axes=0,
    rnn=rnn,
    mark_rnn_endpoint=False,
    camera=cam,
    pca_sample_points=30,
)

# viz.show(scale=.75,     show_rnn_inputs_vectors = False)
# screenshot(f"./paper/images/F3_G_task2.png")


# --------------------------------- run task --------------------------------- #
inputs, outputs = generate_trial()

# evolve RNN dynamics with inputs
h = M.points[40].embedded
rnn.run_initial_condition(h, n_seconds=trial_n_sec, inputs=inputs, cut=False)


# ---------------------------- readout predictions --------------------------- #
# predictions = np.apply_along_axis(rnn.B.T.dot, 1, rnn.traces[-1].trace) # / 7
predictions = np.linalg.pinv(rnn.B / B_scale) @ rnn.traces[-1].trace.T * 2
predictions[0] -= predictions[0, 0]
predictions[1] -= predictions[1, 0]


# ----------------------------- plot task results ---------------------------- #
f, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(9, 9))
colors = (amber_darker, pink_darker)
for n, ax in enumerate(axes):
    ax.plot(
        outputs[:, n],
        lw=2,
        ls="--",
        color=[0.2, 0.2, 0.2],
        label=f"correct output {n}",
    )

    ax.plot(inputs[:, n], lw=2, color=colors[n], label=f"input {n}")
    ax.plot(
        predictions[n, :], lw=3, color=salmon_darker, label=f"prediction {n}"
    )

    ax.legend()
    ax.set(
        ylim=[-2.2, 2.2],
        xticks=np.arange(0, predictions.shape[1] + 1, 30 / RNN.dt),
        xticklabels=np.arange(0, trial_n_sec + 1, 30),
    )

# f, ax = plt.subplots()
# ax.scatter(
#     predictions[0, :], predictions[1], c=np.arange(predictions.shape[1])
# )
# ax.set(xlim=[-1, 1], ylim=[-1, 1], )
clean_axes(f)
plt.show()
f.savefig(f"paper/images/3G_task.svg", format="svg")
