
import matplotlib.pyplot as plt

from manifold.manifolds import Circle
from manifold.topology import Point

from numpy import sin, cos

def embedding(p):
    return Point(
        (sin(p[0]), cos(p[0]), sin(p[0]) + cos(2*p[0]))
    )


S = Circle(
    embedding
)


S.visualize_embedded()

plt.show()