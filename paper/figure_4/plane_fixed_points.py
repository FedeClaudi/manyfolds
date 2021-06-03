import sys

sys.path.append("./")
import numpy as np

# from vedo import screenshot
import matplotlib.pyplot as plt

from myterial import amber_darker, pink_darker, salmon_darker

from manifold import embeddings, Plane
from manifold.rnn import RNN

from manifold import Visualizer
from manifold import visualize

visualize.point_size = 0.03
visualize.reco_surface_radius = 0.0
visualize.rnn_trace_radius = 0.015
visualize.tangent_vector_radius = 0.015
visualize.rnn_inputs_radius = 0.01

N = 128
K = 64
n_inputs = 2
trial_n_sec = 300
SHOW_INPUTS = True

cam = dict(
    pos=(-0.325, -0.171, 5.77),
    focalPoint=(0.0535, -0.0282, 2.81e-3),
    viewup=(0.317, 0.947, 0.0443),
    distance=5.78,
    clippingRange=(5.38, 6.29),
)


def vfield(p):
    s1 = np.sin(2 * np.pi * p[0]) * 0.3
    s2 = np.sin(2 * np.pi * p[1]) * 0.3
    return (-s1, -s2)


# def vfield(p):
#     # s1 = np.sin(np.pi * p[0]) * .8
#     s1 = np.tanh(p[0]) - np.tanh(1 - p[0])
#     s2 = np.tanh(p[1]) - np.tanh(1 - p[1])
#     return (-s1, -s2)


def input_one_vfield(p):
    return (7, 0)


def input_two_vfield(p):
    return (0, 7)


def generate_trial():
    n_secs = trial_n_sec
    n_steps = int(n_secs / RNN.dt)

    inputs = np.zeros((n_steps, 2))
    outputs = np.zeros_like(inputs)

    for i in range(2):
        # ever n sec flip the inputs
        f = 0.1
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
M = Plane(embeddings.prepare_flat_plane_to_rn(n=N), n_sample_points=[8, 8])

M.vectors_field = vfield
M.print_embedding_bounds()

# ---------------------------------- fir RNN --------------------------------- #
rnn = RNN(M, n_units=N, n_inputs=n_inputs)
rnn.build_W(k=K, scale=1)

if SHOW_INPUTS:
    rnn.build_B(k=6, vector_fields=[input_one_vfield, input_two_vfield])
# rnn.run_points(n_seconds=60, cut=False)

# --------------------------------- visualize -------------------------------- #
viz = Visualizer(
    M,
    axes=0,
    rnn=rnn,
    mark_rnn_endpoint=True,
    camera=cam,
    pca_sample_points=30,
)

# --------------------------------- run task --------------------------------- #
inputs, outputs = generate_trial()

# evolve RNN dynamics with inputs
h = M.points[41].embedded
rnn.run_initial_condition(h, n_seconds=trial_n_sec, inputs=inputs, cut=False)

# ------------------------------ visulize in 3D ------------------------------ #
# show(tube, new=True)
# viz.show(scale=0.2, show_rnn_inputs_vectors=True, show_tangents=False)
# screenshot(f"./paper/figure_4/{M.name}_inputs{SHOW_INPUTS}.png")

# ---------------------------- readout predictions --------------------------- #
predictions = np.apply_along_axis(rnn.B.T.dot, 1, rnn.traces[-1].trace) / 7
predictions -= predictions[0]

# TODO improve readout directions
# TODO improve stability

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
        predictions[:, n], lw=3, color=salmon_darker, label=f"prediction {n}"
    )

    ax.legend()
    ax.set(ylim=[-2.2, 4.2])
plt.show()
