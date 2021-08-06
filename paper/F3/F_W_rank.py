import numpy as np
import matplotlib.pyplot as plt
from fcutils.plot.figure import clean_axes
from numpy import cos, sin

import sys

sys.path.append("./")
from manifold import Line
from manifold.manifolds.embeddings import TwoStepsEmbedding
from manifold.manifolds._embeddings import parse, helix_to_r3

from manifold.rnn import RNN

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update(
    {
        "font.size": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)


@parse
def phi_0(p):
    return (p, 0, 0)


@parse
def phi_1(p):
    return (p, sin(p), 0)


@parse
def phi_2(p):
    return (p, sin(p), cos(p))


labels = [
    "$\\phi(p) = (p, 0, 0)$",
    "$\\phi(p) = (p, sin(p), 0)$",
    "$\\phi(p) = (p, sin(p), cos(p))$",
    "$\\phi(p) = (\\frac{cos(4 * \\pi * p)}{2}, \\frac{sin(4 * \\pi * p)}{2}, p + 0.25)$",
]
phis = [phi_0, phi_1, phi_2, helix_to_r3]

f, ax = plt.subplots(figsize=(9, 9))


DIM = (32, 64, 128, 256)
for _phi, lbl in zip(phis, labels):
    data = []
    for N in DIM:
        phi = TwoStepsEmbedding("phi", _phi, N=N)
        M = Line(phi, n_sample_points=4)

        rnn = RNN(M, n_units=N)
        rnn.build_W(k=3, scale=1)

        rnk = np.linalg.matrix_rank(rnn.W)
        data.append(rnk)

    ax.plot(DIM, data, "o-", label=lbl, alpha=0.5)
ax.legend()
ax.set(xlabel="N dimensions", ylabel="rank", xticks=DIM, yticks=[1, 2, 3])
clean_axes(f)

plt.show()
f.savefig(f"paper/images/3F_W_rank.svg", format="svg")
