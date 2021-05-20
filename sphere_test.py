
import matplotlib.pyplot as plt

from manifold.manifolds import Sphere
from manifold.topology import Point

from numpy import sin, cos

def embedding(p):
    return Point(
        (sin(p[0])*cos(p[1]), sin(p[0])*sin(p[1]), cos(p[0])))


S = Sphere(
    embedding,
    n_sample_points = [5, 10]
)


S.visualize_embedded()

plt.show()