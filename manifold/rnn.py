import numpy as np
from loguru import logger
from dataclasses import dataclass
from rich.progress import track

from manifold.tangent_vector import get_tangent_vector


# ------------------------------ helper classes ------------------------------ #
@dataclass
class Trace:
    """
        Stores a trace of the RNN's hidden state over a simulation
    """

    initial_condition: np.ndarray
    trace: np.ndarray

    def __len__(self):
        return len(self.trace)


# ---------------------------------------------------------------------------- #
#                                      RNN                                     #
# ---------------------------------------------------------------------------- #
class RNN:
    dt = 0.05  # simulation dt

    traces = []  # stores results of runnning RNN on initial conditions
    B = None  # place holder for connections matrix

    def __init__(self, manifold=None, n_units=3, n_inputs=None):
        """
            Constructs an RNN objects with the connectivity matrix matching 
            a target manifold.

            h_dot = W sigma(h)
        """
        self.manifold = manifold
        self.n_units = n_units
        self.n_inputs = n_inputs

        self.sigma = np.tanh

        if manifold is not None:
            self.d = self.manifold.d
            self.n = self.manifold.n

    @staticmethod
    def _solve_eqs_sys(Xs, Ys):
        X = np.vstack(Xs).T
        Y = np.linalg.pinv(np.vstack(Ys).T)
        # noise_x = np.random.normal(0, 1e-10, size=X.shape)
        noise = np.random.randn(*Y.shape) * 1e-6
        return X @ (Y + noise)

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
        if self.d == 2:
            # make sure not to oversample
            k = int(np.ceil(np.sqrt(k)))
        points = self.manifold.sample(
            n=k - 1, fill=True, full=self.manifold._full
        )
        if len(points) != k and len(points) != k ** 2:
            raise ValueError(f"Got {len(points)} points with k={k}")

        # get all the vectors
        v = []  # tangent vectors
        s = []  # states through non-linearity
        for point in points:
            # get the network's h_dot as a sum of base function tangent vectors
            vec = get_tangent_vector(point, self.manifold.vectors_field)
            v.append(vec)
            s.append(self.sigma(point.embedded))

        # get W
        self.W = self._solve_eqs_sys(v, s) * self.dt * scale
        logger.debug(f"RNN connection matrix shape: {self.W.shape}")

    def dynamics(self, h):
        return self.W @ self.sigma(h)

    def step(self, h, inputs=None):
        h = np.array(h)
        hdot = self.dynamics(h)

        if inputs is not None:
            hdot += self.B.dot(inputs)
        return h + self.dt * hdot

    def run_initial_condition(
        self, h, n_seconds=10.0, inputs=None, cut=True, cut_th=5
    ):
        trace = [h]
        n_steps = int(n_seconds / self.dt)

        trace = np.zeros((n_steps, len(h)))
        for step in range(n_steps):
            if inputs is not None:
                _step_inputs = inputs[step, :]
            else:
                _step_inputs = None
            h = self.step(h, inputs=_step_inputs)
            trace[step, :] = h

            if np.linalg.norm(h) >= cut_th and step > 1 and cut:
                trace = trace[:step, :]
                break  # too far from origin

        self.traces.append(Trace(trace[0, :], trace))

    def run_points(self, points=None, n_seconds=10, inputs=None, cut=True):
        """
            Runs the RNN on each sampled point for the manifold
        """
        points = points or self.manifold.points

        for point in track(
            points, description="Running initial conditions..."
        ):
            self.run_initial_condition(
                point.embedded, n_seconds=n_seconds, inputs=None, cut=cut,
            )
