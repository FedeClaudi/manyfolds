from pyinspect import install_traceback

install_traceback()

from manifold.manifolds import (
    Line,
    Circle,
    Sphere,
    Plane,
    Torus,
    Cylinder,
)
from manifold.visualize import Visualizer

from vedo import settings

settings.screenshotTransparentBackground = 1
