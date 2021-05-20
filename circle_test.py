import matplotlib.pyplot as plt

from manifold.manifolds import Circle

from numpy import sin, cos


def embedding(p):
    if not isinstance(p, float):
        p = p[0]
    return (sin(p), cos(p), sin(p) + cos(2 * p))


S = Circle(embedding)

# create base functions at each point in the manfiold
S.get_base_functions()

# visualize in embedding
ax = S.visualize_embedded()
S.visualize_base_functions_at_point(ax)


plt.show()
