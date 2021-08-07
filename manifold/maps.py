from numpy import pi

from manifold.decorators import return_many


"""
    Various types of mathematical functions (maps).
    For use as chart maps or basis functions.
"""


def addition(k, *args):
    """
        Base function for addition maps
    """
    out = []
    for x in args:
        if isinstance(x, (tuple, list)) and len(x) > 1:
            out.append([xx + k for xx in x])
        elif isinstance(x, (tuple, list)):
            out.append(x[0] + k)
        else:
            out.append(x + k)
    return out


@return_many
def identity(*args):
    return addition(0, *args)


@return_many
def constant(k, *args):
    out = [k for _ in args]
    return out


@return_many
def subtract_pi(*args):
    return addition(-pi, *args)


@return_many
def subtract_pi_inverse(*args):
    return addition(+pi, *args)


def scalar_multiplication(k, *args):
    """
        Base function for scalar multiplication maps
    """
    out = []
    for x in args:
        try:
            if isinstance(x, (tuple, list)) and len(x) > 1:
                out.append([xx * k for xx in x])
            elif isinstance(x, (tuple, list)):
                out.append(x[0] * k)
            else:
                out.append(x * k)
        except TypeError:
            # x was likely a class instance to be ignored
            continue
    return out


@return_many
def smul_2(*args):
    return scalar_multiplication(2, *args)


@return_many
def smul_2_inverse(*args):
    return scalar_multiplication(0.5, *args)


@return_many
def smul_pi(*args):
    return scalar_multiplication(pi, *args)


@return_many
def smul_pi_inverse(*args):
    return scalar_multiplication(1 / pi, *args)


@return_many
def smul_2pi(*args):
    return scalar_multiplication(2 * pi, *args)


@return_many
def smul_2pi_inverse(*args):
    return scalar_multiplication(1 / (2 * pi), *args)


@return_many
def sphere_U_2(x):
    x0 = x[0] / pi
    x1 = (x[1] - pi) / pi

    return tuple([x0, x1])


@return_many
def sphere_U_2_inverse(x):
    x[:, 0] = x[:, 0] * pi
    x[:, 1] = x[:, 1] * pi + pi

    return x


@return_many
def torus_U_1(x):
    x0 = x[0] / pi
    x1 = x[1] / (2 * pi)

    return tuple([x0, x1])


@return_many
def torus_U_1_inverse(x):
    x[:, 0] = x[:, 0] * pi
    x[:, 1] = x[:, 1] * (2 * pi)

    return x


@return_many
def torus_U_2(x):
    x0 = (x[0] - pi) / pi
    x1 = x[1] / (2 * pi)

    return tuple([x0, x1])


@return_many
def torus_U_2_inverse(x):
    x[:, 0] = x[:, 0] * pi + pi
    x[:, 1] = x[:, 1] * (2 * pi)

    return x


@return_many
def cylinder_U_1(x):
    x1 = x[1]
    x0 = x[0] / pi
    return tuple([x0, x1])


@return_many
def cylinder_U_1_inverse(x):
    x[:, 0] = x[:, 0] * pi
    return x


@return_many
def cylinder_U_2(x):
    x1 = x[1]
    x0 = (x[0] - pi) / pi
    return tuple([x0, x1])


@return_many
def cylinder_U_2_inverse(x):
    x[:, 0] = pi * x[:, 0] + pi
    return x
