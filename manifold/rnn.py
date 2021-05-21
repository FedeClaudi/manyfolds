import numpy as np
from loguru import logger
from dataclasses import dataclass

from manifold.maths import min_distance_from_point


@dataclass
class Trace:
    """
        Stores the results of running the RNN on one initial condition
    """

    initial_condition: np.ndarray
    trace: np.ndarray


class RNN:
    dt = 0.001
    sigma = np.tanh

    traces = []  # stores results of runnning RNN on initial conditions

    def __init__(self, manifold, n_units=3):
        """
            Constructs an RNN objects with the connectivity matrix matching 
            a target manifold.

            h_dot = W sigma(h)
        """
        self.manifold = manifold
        self.n_units = n_units

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
        # sample points
        points = self.manifold.sample(n=k + 2, fill=True)[1:-1]

        # get all the vectors
        v = []  # tangent vectors
        s = []  # states through non-linearity
        for n, point in enumerate(points):
            v.append(point.base_functions[0].tangent_vector * scale)
            s.append(self.sigma(point.embedded))

        V = np.linalg.pinv(np.vstack(v).T)
        S = np.linalg.pinv(np.vstack(s).T)
        """
            Note:
                from the derivation we get:
                    Hdot = W * H
                but to solve for W we need to have:
                    AW = B
                so we multiply things around to get
                    H^-1 * W = Hdot^-1
                and solve for W
        """

        # get W
        self.W = np.linalg.lstsq(V, S, rcond=None)[0]
        # self.W = np.linalg.solve(V, S)
        logger.debug(f"RNN connection matrix:\n {self.W}")

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

        logger.debug(
            f"Initial condition completed - trace ends at distance from manifold: {min_distance_from_point(self.manifold.embedded, trace[-1, :])}"
        )

    def run_points(self, points=None, n_seconds=10):
        """
            Runs the RNN on each sampled point for the manifold
        """
        points = points or self.manifold.points

        for point in points:
            self.run_initial_condition(
                np.array(point.embedded), n_seconds=n_seconds
            )

    def plot_traces(self, ax):
        for trace in self.traces:
            ax.plot(
                trace.trace[:, 0],
                trace.trace[:, 1],
                trace.trace[:, 2],
                c="green",
                lw=2,
            )
