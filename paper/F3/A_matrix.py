from loguru import logger

import matplotlib.pyplot as plt
from fcutils.plot.figure import clean_axes

import sys

sys.path.append("./")
from manifold import embeddings, Line, Circle
from manifold.rnn import RNN


from manifold import visualize

visualize.rnn_trace_alpha = 0.6
visualize.reco_surface_radius = 0.1

"""
    Plots the connectivity matrix of networks fitted to different manifolds. 
"""

# --------------------------------- settings --------------------------------- #
MANIFOLD = "helix"
N = 64
K = 12

# ---------------------------------------------------------------------------- #
#                          select manifold parameters                          #
# ---------------------------------------------------------------------------- #
cam = None
# get manifold
if MANIFOLD == "line":
    logger.debug("Line manifold")
    M = Line(embeddings.line_to_rn, n_sample_points=1)
    pca_sample_points = 50

    if K > 6:
        logger.warning("Line manifold prefers K = 6")

    cam = dict(
        pos=(-1.05, 0.545, 5.27),
        focalPoint=(0.0378, 0.0910, 5.63e-3),
        viewup=(0.756, 0.646, 0.101),
        distance=5.39,
        clippingRange=(4.62, 6.38),
    )

elif MANIFOLD == "helix":
    logger.debug("helix manifold")
    M = Line(embeddings.helix_to_rn, n_sample_points=4)
    pca_sample_points = 150

    if K > 12 or K < 8:
        logger.warning("Helix manifold prefers K = 10")

elif MANIFOLD == "circle":
    logger.debug("Circle manifold")
    M = Circle(embeddings.circle_to_rn_flat, n_sample_points=2)

    pca_sample_points = 50
    # M.vectors_field = vectors_fields.double_sin

    cam = dict(
        pos=(-5.18, 5.94, 1.54),
        focalPoint=(-0.122, -0.0158, 0.0684),
        viewup=(0.154, -0.112, 0.982),
        distance=7.95,
        clippingRange=(5.30, 11.5),
    )


# ---------------------------------------------------------------------------- #
#                              stuff happens here                              #
# ---------------------------------------------------------------------------- #


# fit and run RNN
rnn = RNN(M, n_units=N)
rnn.build_W(k=K, scale=1)
rnn.run_points(n_seconds=15, cut=True)


f, ax = plt.subplots(figsize=(9, 9))

ax.imshow(rnn.W, cmap="bwr")
ax.set(xticks=[], yticks=[])
clean_axes(f)

plt.show()
f.savefig(f"paper/images/3C_{MANIFOLD}_mtx.svg", format="svg")
