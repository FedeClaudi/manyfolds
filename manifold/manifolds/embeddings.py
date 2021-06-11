import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import diff, lambdify
from sympy.abc import p


from manifold.topology import Point


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

        self.phi_star = []
        for part in self._phi:
            d1 = part.diff("p0")
            d2 = part.diff("p1")
            self.phi_star.append(lambdify(["p0", "p1"], d1 + d2))

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

        # res = np.vstack([der(coords[:, 0], coords[:, 1]) for der in self.phi_star]).T
        # return np.diff(res, axis=0)[50] * 1000
        return np.hstack([der(*at_point) for der in self.phi_star])

    def __repr__(self):
        return f'Embedding: {self.name} | phi: "{self._phi}" | phi_star: "{self._phi_star}"'

    def __str__(self):
        return self.__repr__()


# # ---------------------------------------------------------------------------- #
# #                                    R^N = 3                                   #
# # ---------------------------------------------------------------------------- #

# ----------------------------------- line ----------------------------------- #
line_to_r3_flat = Embedding("line_to_r3_flat", "p, p, p")
line_to_r3 = Embedding(
    "line_to_r3", "sin(2 * p) - 0.5, sin(p) * 2 - 1, -cos(p) * 4 + 3"
)
helix_to_r3 = Embedding(
    "helix_to_r3", "cos(4 * pi * p) / 2, sin(4 * pi * p) / 2, p + 0.25"
)

# ---------------------------------- circle ---------------------------------- #
circle_to_r3_angled = Embedding(
    "circle_to_r3_angled", "sin(p), cos(p), sin(p)"
)
circle_to_r3_bent = Embedding(
    "circle_to_r3_bent", "sin(p), 0.8 * cos(p), cos(p) ** 2 * 0.5 + 0.5"
)
circle_to_r3 = Embedding(
    "circle_to_r3", "sin(p), 0.8 * cos(p), cos(p * 2) ** 2 * 0.5 + 0.5"
)


# ----------------------------------- plane ---------------------------------- #

plane_to_r3_flat = Embedding2D(
    "plane_to_r3_flat", "p0 + 0.2, p1 + 0.2, 0.5 * (p0 + p1)"
)
plane_to_r3 = Embedding2D(
    "plane_to_r3", "p0, sin(p1) * 2, 0.4 * (p1 - p0) ** 2"
)


# ---------------------------------- sphere ---------------------------------- #
sphere_to_r3 = Embedding2D(
    "sphere_to_r3", "sin(p0) * cos(p1), sin(p0) * sin(p1), cos(p0)"
)

# ----------------------------------- torus ---------------------------------- #
torus_to_r3 = Embedding2D(
    "torus_to_r3",
    "(.75 + .25 * cos(p0)) * cos(p1), (.75+ .25 * cos(p0)) * sin(p1), .25 * sin(p0)",
)


# def torus_to_r3(p0, p1):
#     R = 0.75  # torus center -> tube center
#     r = 0.25  # tube radius
#     return (
#         (R + r * cos(p0)) * cos(p1),
#         (R + r * cos(p0)) * sin(p1),
#         r * sin(p0),
#     )

# --------------------------------- cylinder --------------------------------- #
cylinder_to_r3 = Embedding2D(
    "cylinder_to_r3", "sin(p0) / 2, cos(p0) / 2, p1 + 0.1"
)

cylinder_to_r3_as_cone = Embedding2D(
    "cylinder_to_r3_as_cone",
    "p1 / 2 + 0.4 * sin(p0) / 2, p1 / 2 + 0.4 * cos(p0) / 2, p1 + 0.5",
)
