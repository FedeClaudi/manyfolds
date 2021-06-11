from sympy.parsing.sympy_parser import parse_expr

helix_to_r3 = parse_expr("cos(4 * pi * p) / 2, sin(4 * pi * p) / 2, p + 0.25")


# # ---------------------------------------------------------------------------- #
# #                                    R^N = 3                                   #
# # ---------------------------------------------------------------------------- #

# # ----------------------------------- line ----------------------------------- #
# @parse
# def line_to_r3_flat(p):
#     return (p, p, p)


# @parse
# def line_to_r3(p):
#     return (sin(2 * p) - 0.5, sin(p) * 2 - 1, -cos(p) * 4 + 3)


# @parse
# def helix_to_r3(p):
#     return (cos(4 * pi * p) / 2, sin(4 * pi * p) / 2, p + 0.25)


# # ---------------------------------- circle ---------------------------------- #
# # @parse
# # def circle_to_r3_flat(p):
# #     """
# #         Embedds a circle in 3D but keeping the circle flat in one dimension
# #     """
# #     return (sin(p), cos(p), 1)


# @parse
# def circle_to_r3_angled(p):
#     return (sin(p), cos(p), sin(p))


# @parse
# def circle_to_r3_bent(p):
#     return (sin(p), 0.8 * cos(p), cos(p) ** 2 * 0.5 + 0.5)


# @parse
# def circle_to_r3(p):
#     return (sin(p), 0.8 * cos(p), cos(p * 2) ** 2 * 0.5 + 0.5)


# # ---------------------------------- sphere ---------------------------------- #
# @parse2D
# def sphere_to_r3(p0, p1):
#     return (sin(p0) * cos(p1), sin(p0) * sin(p1), cos(p0))


# @parse2D
# def ellipse_to_r3(p0, p1):
#     return (sin(p0) * cos(p1) * 0.3, sin(p0) * sin(p1) * 0.3, cos(p0))


# # ----------------------------------- plane ---------------------------------- #
# @parse2D
# def plane_to_r3_flat(p0, p1):
#     return (p0 + 0.2, p1 + 0.2, 0.5 * (p0 + p1))


# @parse2D
# def plane_to_r3(p0, p1):
#     return (p0, sin(p1) * 2, 0.4 * (p1 - p0) ** 2)


# # ----------------------------------- torus ---------------------------------- #
# @parse2D
# def torus_to_r3(p0, p1):
#     R = 0.75  # torus center -> tube center
#     r = 0.25  # tube radius
#     return (
#         (R + r * cos(p0)) * cos(p1),
#         (R + r * cos(p0)) * sin(p1),
#         r * sin(p0),
#     )


# @parse2D
# def thin_torus_to_r3(p0, p1):
#     R = 1  # torus center -> tube center
#     r = 0.18  # tube radius
#     return (
#         (R + r * cos(p0)) * cos(p1),
#         (R + r * cos(p0)) * sin(p1),
#         r * sin(p0),
#     )


# # --------------------------------- cylinder --------------------------------- #
# @parse2D
# def cylinder_to_r3(p0, p1):
#     return (sin(p0) / 2, cos(p0) / 2, p1 + 0.1)


# @parse2D
# def cylinder_to_r3_as_cone(p0, p1):
#     k = p1 / 2 + 0.4
#     return (k * sin(p0) / 2, k * cos(p0) / 2, p1 + 0.5)
