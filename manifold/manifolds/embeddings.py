from numpy import sin, cos


def floatify(func):
    def inner(p):
        if not isinstance(p, float):
            p = p[0]
        return func(p)

    return inner


# ----------------------------------- line ----------------------------------- #
@floatify
def line_to_r3_flat(p):
    return (p, p, p)


@floatify
def line_to_r3(p):
    return (sin(2 * p), sin(p), -cos(p))


# ---------------------------------- circle ---------------------------------- #
@floatify
def circle_to_r3_flat(p):
    """
        Embedds a circle in 3D but keeping the circle flat in one dimension
    """
    return (sin(p) / 6, cos(p) / 6, 1)
