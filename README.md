Code for the paper: **Differential geometry methods for constructing manifold-targeted recurrent neural networks.**

### Introduction
The code used to generate the figures in the manuscript is in the folder `paper` and is organized by figure number with (generally) one script
for each figure panel (except those that were created not programmatically like illustrations etc.).

The code on the `paper` folder relies on the python package whose code is in the `manifold` folder.
`Manifold` includes code to deal with topological manifold and embeddings, construct RNNs targeted to a manifold (as described in the paper) and visualize
both manifolds and RNN dynamics with 3D interactive visualizations (implemented in `vedo`).


### Code organization
Scripts starting with a single underscore (`_x.py`) container helper function used in files without the underscore (`x.py`).

* `base_function.py` defines two classes that represent base functions (in the chart representation of a manifold) and their projection on the embedded manifold.
* `embedding_functions.py` is a collection of functions of the form $\phi(p): \mathcal M \to \mathbb R^d$ representing embedding of (points on) manifolds in a d-dimensional Euclidean space.
* `maps.py` defines a collection of functions $x(p) \mathcal M \to \mathbb R^d$ used as chart maps for the charts defined on the manifolds
* `rnn.py` defines the `RNN` class which handles the creation of an RNN fitted to a target manifold and it's dynamics (simulations from an initial condition)
* `tangent_vector.py` has code that helps with the computation of tangent vectors on embedded manifolds.
* `topology.py` containts classes for the abstraction of topological objects (`Manifold`, `Chart`, `Point`)
* `vector_fields.py` containts a list of functions $\psi(p): \mathcal M \to \mathbb R^d$ used to define vector fields of tangent vectors on the topological manifold.
* `visualize.py` has the `Visualizer` class which handles the rendering in 3D of manifolds and RNN dynamics (after dimensionality reduction with PCA, if necessary).


The folder `manifolds` contain two scripts for the definition of one dimensional and two dimensional manifolds. Each manifold is defined by a name, a set $M$, a set of charts and associated base functions.  The definition of the set, charts and chat maps relies on the classes defined in `topology.py` and maps from `maps.py`. E.g. the sphere is defined as:

```python

class Sphere(Manifold2D):
    name = "S^2"
    manifold = Manifold(
        M=[Interval("M_1", 0, pi), Interval("M_2", 0, 2 * pi)],
        charts=[
            Chart(
                1,
                [Interval("U_1_1", 0, pi), Interval("U_1_2", 0, pi)],
                Map("x_1", maps.smul_pi_inverse, maps.smul_pi),
            ),
            Chart(
                2,
                [Interval("U_2_1", 0, pi), Interval("U_2_2", pi, 2 * pi)],
                Map("x_2", maps.sphere_U_2, maps.sphere_U_2_inverse),
            ),
        ],
    )

```

All manifold classes inherit from abstract classes `BaseManifold>Manifold1D` and `BaseManifold>Manifold2D` which have methods to sample points on the manifolds and to handle charts and basis functions.

## Usage
### Creating/embedding/visualizing manifolds
This example shows how to create a `Plane` manifold, embed it into $\mathbb R^3$, define a tangent vector field on it and compute tangent vectors, finally the manifold is visualized in 3D.

```python 

import sys
import numpy as np
sys.path.append("./")

from manifold import Plane, Visualizer
from manifold._embeddings import Embedding
from manifold.decorators import parse2D
import manifold

# set some parameters to make the visualization look nicer
manifold.visualize.reco_surface_radius = 0.05 * 3
manifold.visualize.point_size = 0.035
manifold.tangent_vector_radius = 0.001
cam = dict(  # camera position parameters
    pos=(0.596, -5.06, 2.41),
    focalPoint=(1.46, 0.435, 0.511),
    viewup=(0.100, 0.311, 0.945),
    distance=5.87,
    clippingRange=(2.97, 12.0),
)

# define an embedding of the plane to R^3, map: p \in R^2 \to R^3
@parse2D
def _curvy_plane_3d(p0, p1):
    return (
        p0 * 3,
        p1 * 3,
        0.3
        - np.sin(1.5 * np.pi * p0)
        * np.sin(1.5 * np.pi * p1)
        * (0.35 - 0.3 * p0)
        * 3,
    )

# define a vector field on the plane manifold
def vfield(point):
    return (
        np.sin(point[0] * np.pi * 2),
        np.sin(point[1] * np.pi * 2),
    )

# create an embedding object with the custom embedding function
plane_3d_embedding = Embedding("plane 3d", _curvy_plane_3d)

# create an instance of the Plane manifold with our custom embedding object
M = Plane(plane_3d_embedding, n_sample_points=16)

# set the vector field
M.vectors_field = vfield

# create an instance of the visualizer object
viz = Visualizer(M, manifold_alpha=1, wireframe=False)

# show rendering
viz.show(
    x_range=[0.1, 0.2],
    scale=0.1,
    show_basis_vecs=False,
    show_tangents=True,
    show_points=False,
    axes=0,
    camera=cam,
)
```


Note that nowhere are we explicitly asking for the tangent vectors to be created, that's done automatically by `Visualizer` in order to produce the visualization.


### Fitting RNN
Fitting an RNN is very simple, given a manifold `M` (e.g. `M = Plane(plane_3d_embedding)` from above), you can create a manifold targeted RNN:

```python
from manifold.rnn import RNN

# create RNN
rnn = RNN(M, n_units=64)  # note that the number of units should match the dimensionality of the embedding space

# fit RNN and compute connectivity matrix W
rnn.build_W(k=K, scale=1)

# simulate RNN dynamics from a number of initial conditoins on the embedded manifold
rnn.run_points()


# If you want to visualize the RNN dynamics alongside your manifold M:
viz = Visualizer(
    M, rnn=rnn,
)
viz.show(x_range=[0.1, 0.2], scale=0.2, axes=0, camera=cam)
```

Two important notes:
    - to fit RNNs to the manifold's tangent vectors, you want the manifold to be embedded in $\mathbb R^n$ (n=64 by default).  In `embeddings.py` a number of embedding function are defined for each manifold. Those with names ending in `r3` are for $mathbb R^3$ and those ending in `rn` and for $\mathbb R^n$. If you want to create an embedding to $\mathbb R^n$, you should use the `TwoStepsEmbedding` to permorm the two step embedding procedure outlined in the manuscript.
    - RNNs are fitted by least squares using constraints imposed by the tangent vectors. This does not guarantee that the RNN dynamics are accurately targeted to the manifold. The accuracy depends on: 1) number of points used to fit the RNN (i.e. number of points sampled on the manifold and used to compute the tangent vectors at those points) and 2) the required dynamics: RNNs can't produce any arbitrary dynamics. If the embedding or vector field are 'bad' the RNN won't produce accurate dynamics (what 'bad' is depends on the RNN architecture, here generally you want the dynamics to by symmetric around the origin and have even number of fixed points).


