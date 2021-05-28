import numpy as np
from loguru import logger
from dataclasses import dataclass
from rich.progress import track

from manifold.maths import tanh, unit_vector
from manifold.tangent_vector import get_tangent_vector


@dataclass
class Trace:
    """
        Stores the results of running the RNN on one initial condition
    """

    initial_condition: np.ndarray
    trace: np.ndarray


@dataclass
class InputsBase:
    idx: int
    vec: np.ndarray
    projected: np.ndarray  # Bu -> basis vec projected in state space


class RNN:
    dt = 0.003
    sigma = tanh

    traces = []  # stores results of runnning RNN on initial conditions
    B = None  # place holder for connections matrix

    def __init__(self, manifold, n_units=3, n_inputs=None):
        """
            Constructs an RNN objects with the connectivity matrix matching 
            a target manifold.

            h_dot = W sigma(h)
        """
        self.manifold = manifold
        self.n_units = n_units
        self.n_inputs = n_inputs

    @staticmethod
    def _solve_eqs_sys(Xs, Ys):
        X = np.vstack(Xs).T
        Y = np.linalg.pinv(np.vstack(Ys).T)
        noise = np.random.normal(0, 1e-6, size=Y.shape)
        return X @ (Y + noise)

    def build_B(self, k=10, vector_fields=None):
        """
            Builds the input connectivity matrix B for the RNN by
            enforinc that any import vector u results in a vector
            tangent to the manifold when multiplied by B.

            Arguments:
                k: int. Number of sample points to use.
                vector_fields: list. List of functions mapping points on the manifolds
                    to elements of the tangent vector spce at that point.
        """
        if vector_fields is not None:
            if isinstance(vector_fields, list) and len(vector_fields) != self.n_inputs:
                raise ValueError(
                    "When passing vector fields to build_B you need as manu fields as there are inputs to the RNN"
                )
            elif not isinstance(vector_fields, list):
                vector_fields = [vector_fields] * self.n_inputs

        points = self.manifold.sample(n=k - 1, fill=True, full=True)

        # basis of inputs vector space Rm
        basis = np.eye(self.n_inputs)
        inputs = []  # stroes inputs vectors
        tangents = []  # stores tangent vectors Bu
        for point in points:
            for idx in range(self.n_inputs):
                # get vectors field
                if vector_fields is not None:
                    vfield = vector_fields[idx]
                else:
                    vfield = None
                # get a vector tangent to the manifold
                tangents.append(
                    get_tangent_vector(point, vectors_field=vfield)
                )

                # get the basis vector
                inputs.append(basis[:, idx])

        # solve for B
        self.B = self._solve_eqs_sys(inputs, tangents)
        logger.debug(f"RNN input matrix shape: {self.B.shape}")
        self.inputs_basis = [
            InputsBase(n, basis[:, n], unit_vector(self.B.T.dot(basis[:, n])))
            for n in range(self.n_inputs)
        ]

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
        points = self.manifold.sample(n=k - 1, fill=True, full=True)
        if len(points) != k and len(points) != k ** 2:
            raise ValueError(f"Got {len(points)} points with k={k}")
        # logger.debug([p.coordinates for p in points])

        # get all the vectors
        v = []  # tangent vectors
        s = []  # states through non-linearity
        for point in points:
            # get the network's h_dot as a sum of base function tangent vectors
            vec = get_tangent_vector(point, self.manifold.vectors_field)

            # keep track for each point to build sys of equations
            v.append(vec)
            s.append(self.sigma(point.embedded))

        # get W
        self.W = self._solve_eqs_sys(v, s) / self.dt * scale
        logger.debug(f"RNN connection matrix shape: {self.W.shape}")

    def step(self, h, inputs=None):
        h = np.array(h)
        if inputs is None:
            hdot = self.W.dot(self.sigma(h))
        else:
            if self.B is None:
                raise ValueError(
                    "In order to use inputs you need to build B matrix first"
                )
            hdot = self.W.dot(self.sigma(h)) + self.B.T.dot(inputs)
        return h + self.dt * hdot

    def run_initial_condition(self, h, n_seconds=10.0, inputs=None):
        trace = [h]
        n_steps = int(n_seconds / self.dt)

        trace = np.zeros((n_steps, len(h)))
        for n, step in enumerate(range(n_steps)):
            h = self.step(h, inputs=inputs)
            trace[n, :] = h

            if np.linalg.norm(h) >= 3 and n > 1:
                trace = trace[:n, :]
                break  # too far from origin

        self.traces.append(Trace(trace[0, :], trace))

    def run_points(self, points=None, n_seconds=10, inputs=None):
        """
            Runs the RNN on each sampled point for the manifold
        """
        points = points or self.manifold.points

        for point in track(
            points, description="Running initial conditions..."
        ):
            self.run_initial_condition(
                np.array(point.embedded), n_seconds=n_seconds, inputs=None
            )
