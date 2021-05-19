import numpy as np
import matplotlib.pyplot as plt

color = '#c3c3db'

class Base:

    def __init__(self, embedding, n_sample_points = 10):
        self.n_sample_points = n_sample_points
        self.embedding = embedding
        self.embedd()

    @property
    def d(self):
        '''
            Dimensionality of the manifold
        '''
        raise NotImplementedError

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

    def sample(self, n=None):
        '''
            Samples N points from the manifold's interval
        '''
        n = n or self.n_sample_points
        return np.linspace(self.M.l, self.M.r, n)

    def embedd(self):
        '''
            Embed N randomly sampled points of the manifold's set
            using the embedding function
        '''
        self.embedded_points = [self.embedding(p) for p in self.sample()]

        self.embedded_points_vis = [self.embedding(p) for p in self.sample(n=200)]


    def visualize_embedded(self):
        if self.n != 2:
            raise NotImplementedError('Visualization only working for 2D')

        f, ax = plt.subplots(figsize=(9, 9))

        for p in self.embedded_points:
            ax.scatter(*p.coordinates, s=500, color=color, zorder=2)

        ax.plot(
            [p.coordinates[0] for p in self.embedded_points_vis],
            [p.coordinates[1] for p in self.embedded_points_vis],
            lw=1.5,
            color=[.7, .7, .7],
            zorder=-1
        )