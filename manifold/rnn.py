import numpy as np
from loguru import logger
from dataclasses import dataclass
from rich.progress import track
from vedo import Tube

from myterial import salmon, salmon_darker

from manifold.maths import tanh
from manifold.visualize import make_palette


@dataclass
class Trace:
    """
        Stores the results of running the RNN on one initial condition
    """

    initial_condition: np.ndarray
    trace: np.ndarray


class RNN:
    dt = 0.003
    sigma = tanh

    traces = []  # stores results of runnning RNN on initial conditions

    def __init__(self, manifold, n_units=3):
        """
            Constructs an RNN objects with the connectivity matrix matching 
            a target manifold.

            h_dot = W sigma(h)
        """
        self.manifold = manifold
        self.n_units = n_units

    # def get_target_jacobian_at_point(self, point, tangent_vector):
    #     '''
    #         To define the target Jacobian's SVD we need an orthogonal
    #         matrix of eigenvectors U and a diagonal matrix of eigenvalues S.

    #         To construct U we get the rotation matrix that rotates the
    #         standard basis of Rn to be aligned to the tangent vector at the point.
    #         Then we use that on the matrix representing the standard basis and
    #         we have our matrix of eigenvectors
    #     '''
    #     # stanrd basis
    #     U = np.eye(len(tangent_vector))
    #     U[:, 0] = tangent_vector

    #     S = np.eye(len(tangent_vector)) *  (-1 / self.dt)
    #     S[0, 0] = 1
    #     return U, S

    def build_W(self, k=10, scale=1):
        """
            For each sample point on the manifold we have 
            h_dot = W sigma(h)

            with h_dot being the tangent vector at the embedded manifold h.
            More precisely, h_dot is a linear combination of the bases
            of a vector space at the embedded point which are computed by
            taking the derivative of the embedding of base functions defined on
            the image of the chart map of a chart containing the point.

            This can be used to find W such that hdot is in the tangent space
            at each point.
        """
        logger.debug(f"RNN - building connectivity matrix with {k} points")

        # sample points
        points = self.manifold.sample(n=k - 1, fill=True,)
        if len(points) != k and len(points) != k ** 2:
            raise ValueError(f"Got {len(points)} points with k={k}")
        # logger.debug([p.coordinates for p in points])

        # get all the vectors
        v = []  # tangent vectors
        s = []  # states through non-linearity
        for n, point in enumerate(points):
            # get the network's h_dot as a sum of base function tangent vectors
            weights = self.manifold.vectors_field(point)

            bases = np.vstack(
                [
                    fn.tangent_vector * w
                    for w, fn in zip(weights, point.base_functions)
                ]
            )
            vec = np.sum(bases, 0)
            # keep track for each point to build sys of equations
            v.append(vec)
            s.append(self.sigma(point.embedded))

        # get W
        V = np.vstack(v).T
        S = np.linalg.pinv(np.vstack(s).T)
        noise = np.random.uniform(0, 1e-6, size=S.shape)
        self.W = V @ (S + noise) / self.dt * scale

        # self.W = np.linalg.solve(V, S)
        logger.debug(f"RNN connection matrix shape: {self.W.shape}")

    def step(self, h):
        h = np.array(h)
        return h + self.dt * (self.W.dot(self.sigma(h)))

    def run_initial_condition(self, h, n_seconds=10.0):
        trace = [h]
        n_steps = int(n_seconds / self.dt)

        trace = np.zeros((n_steps, len(h)))
        for n, step in enumerate(range(n_steps)):
            h = self.step(h)
            trace[n, :] = h

        self.traces.append(Trace(trace[0, :], trace))

    def run_points(self, points=None, n_seconds=10):
        """
            Runs the RNN on each sampled point for the manifold
        """
        points = points or self.manifold.points

        for point in track(
            points, description="Running initial conditions..."
        ):
            self.run_initial_condition(
                np.array(point.embedded), n_seconds=n_seconds
            )

    def plot_traces(self, skip=1):
        colors = make_palette(salmon_darker, salmon, len(self.traces[0].trace))
        for trace in self.traces:
            self.manifold.actors.append(Tube(trace.trace, c=colors, r=0.005,))
