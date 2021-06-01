from pyinspect import install_traceback

install_traceback()

from manifold.manifolds import (
    embeddings,
    Line,
    Circle,
    Sphere,
    Plane,
    Torus,
    vectors_fields,
    Cylinder,
)
from manifold.visualize import Visualizer

from vedo import settings

settings.screenshotTransparentBackground = 1
