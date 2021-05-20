import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from manifold.topology import Point, Interval, Interval2D

color = '#c3c3db'
grey = [.6, .6, .6]


class Base:
    def __init__(self, embedding, n_sample_points =10):
        self.n_sample_points = n_sample_points
        self.embedding = embedding
        self.embedd()

    @property
    def d(self):
        '''
            Dimensionality of the manifold
        '''
        if isinstance(self.manifold.M, Interval):
            return 1
        elif isinstance(self.manifold.M, Interval2D):
            return 2
        else:
            raise NotImplementedError('Cant deal with manifold of dimension > 2')

    @property
    def n(self):
        '''
            Dimensionality of the embedding
        '''
        return len(self.embedded_points[0].coordinates)

    @property
    def M(self):
        ''' short hand for self.manifold.M '''
        return self.manifold.M

    def sample(self, n=None, filter=True, tol=1e-1):
        '''
            Samples N points from the manifold's interval ensuring that they are not too close
        '''

        # how many points
        n = n or self.n_sample_points
        if not isinstance(n, list):
            n = [n] * self.d
        logger.debug(f'Sampling manifold points with n={n}')

        # sample points
        if self.d == 1:
            points = [Point((k,)) for k in np.linspace(self.M.l, self.M.r, n[0])]
        elif self.d == 2:
            points = []
            for p1 in np.linspace(self.manifold.M.int1.l, self.manifold.M.int1.r, n[0]):
                for p2 in np.linspace(self.manifold.M.int2.l, self.manifold.M.int2.r, n[1]):
                    points.append(Point((p1,p2)))
        logger.debug(f'Sampled {len(points)} points, filtering...')

        # discard points too close to other points
        if filter:
            selected = []
            for p in points:
                distances = [np.linalg.norm(p.acord - s.acord) for s in selected]
                if not np.any(np.array(distances) <= tol):
                    selected.append(p)
        else:
            selected = points

        logger.debug(f'Finished sampling with {len(selected)} points left')
        return selected

    def embedd(self):
        '''
            Embed N randomly sampled points of the manifold's set
            using the embedding function
        '''
        self.embedded_points = [self.embedding(p) for p in self.sample()]
        self.embedded_points_vis = [self.embedding(p) for p in self.sample(n=[50, 200], filter=False)]


    def _visualize_embedding_2D(self):
        f, ax = plt.subplots(figsize=(9, 9))

        for p in self.embedded_points:
            ax.scatter(*p.coordinates, s=200, color=color, zorder=2, edgecolors=grey, lw=.5)

        ax.plot(
            [p.coordinates[0] for p in self.embedded_points_vis],
            [p.coordinates[1] for p in self.embedded_points_vis],
            lw=1.5,
            color=grey,
            zorder=-1
        )

    def _visualize_embedding_3D(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for p in self.embedded_points:
            ax.scatter(*p.coordinates, s=200, color=color, zorder=2, edgecolors=grey, lw=.5)

        ax.plot(
            [p.coordinates[0] for p in self.embedded_points_vis],
            [p.coordinates[1] for p in self.embedded_points_vis],
            [p.coordinates[2] for p in self.embedded_points_vis],
            lw=1.5,
            color=grey,
            zorder=-1
        )

    def visualize_embedded(self):
        if self.n == 2:
            self._visualize_embedding_2D()
        elif self.n == 3:
            self._visualize_embedding_3D()
        else:
            raise NotImplementedError(f'Unrecognized number of dimensions: {self.n}')

