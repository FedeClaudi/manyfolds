import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import diff, lambdify
from sympy.abc import p


from manifold.topology import Point
from manifold.maths import unit_vector


class Embedding:
    def __init__(self, name, expression):
        self.name = name
        self._phi = parse_expr(expression)
        self._phi_star = diff(self._phi, p)

        self.phi = lambdify(p, self._phi)
        self.phi_star = [lambdify(p, part.diff()) for part in self._phi]

    def __call__(self, val):
        if isinstance(val, Point):
            val = val[0]
        return self.phi(val)

    def push_forward(self, base_function):
        coords = base_function.get_manifold_coordinates()
        at_point = (
            coords[base_function.embedded_point_index, :]
            .ravel()
            .astype(np.float64)
        )
        return np.hstack([der(at_point) for der in self.phi_star])

    def __repr__(self):
        return f'Embedding: {self.name} | phi: "{self._phi}" | phi_star: "{self._phi_star}"'

    def __str__(self):
        return self.__repr__()


class Embedding2D:
    def __init__(self, name, expression):
        self.name = name
        self._phi = parse_expr(expression)
        self._phi_star = diff(self._phi, "p0", "p1")

        self.phi = lambdify(["p0", "p1"], self._phi)

        self.phi_star_0, self.phi_star_1 = [], []
        for part in self._phi:
            self.phi_star_0.append(lambdify(["p0", "p1"], part.diff("p0")))
            self.phi_star_1.append(lambdify(["p0", "p1"], part.diff("p1")))

    def __call__(self, val):
        if isinstance(val, Point):
            p0, p1 = val.coordinates
        else:
            p0, p1 = val
        return self.phi(p0, p1)

    def push_forward(self, base_function):
        if isinstance(base_function, np.ndarray):
            at_point = base_function
            raise ValueError
        else:
            coords = base_function.manifold_coords
            at_point = (
                coords[base_function.embedded_point_index, :]
                .ravel()
                .astype(np.float64)
            )

        partial_derivative = (
            self.phi_star_0 if base_function.dim_idx == 0 else self.phi_star_1
        )
        return unit_vector(
            np.hstack([der(*at_point) for der in partial_derivative])
        )

    def __repr__(self):
        return f'Embedding: {self.name} | phi: "{self._phi}" | phi_star: "{self._phi_star}"'

    def __str__(self):
        return self.__repr__()
