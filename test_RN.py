import matplotlib.pyplot as plt
from loguru import logger

from manifold import embeddings, Line, Circle, Torus
from manifold.highN import Visualizer
from manifold.rnn import RNN
from manifold import vectors_fields

MANIFOLD = "circle"

# get manifold
if MANIFOLD == "line":
    logger.debug("Line manifold")
    M = Line(embeddings.prepare_line_to_rn(n=64), n_sample_points=4)
    remove_ax_lims = True
elif MANIFOLD == "circle":
    logger.debug("Circle manifold")
    M = Circle(embeddings.prepare_circle_embedding(n=64), n_sample_points=8)
    remove_ax_lims = False
elif MANIFOLD == "torus":
    logger.debug("Torus manifold")
    M = Torus(embeddings.prepare_torus_to_rn(n=64), n_sample_points=[4, 4])
    remove_ax_lims = True
else:
    raise NotImplementedError

# set vector field
M.vectors_field = vectors_fields.sin


# fit and run RNN
rnn = RNN(M, n_units=64)
rnn.build_W(k=64)
rnn.run_points(n_seconds=5)


viz = Visualizer(M, rnn=None, pca_sample_points=20)
ax = viz.show(ax_lims=remove_ax_lims, scale=0.2)

# TODO check tangent vectors correctly plotted

plt.show()
