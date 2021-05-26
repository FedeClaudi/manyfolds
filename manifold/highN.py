from sklearn.decomposition import PCA
import numpy as np
from vedo import show, Spheres, Tube, recoSurface

from myterial import grey, blue, salmon, green, grey_dark

from manifold.topology import Point


class Visualizer:
    """
        Class to facilitate the visualization of 
        data from embeddings in N >> 3 using 
        dimensionality reduction techniques.
    """

    actors = []

    def __init__(self, manifold, rnn=None, pca_sample_points=200):
        self.pca_sample_points = pca_sample_points
        self.manifold = manifold
        self.rnn = rnn
        self.fit_pca()

    def fit_pca(self):
        """
            Fits a PCA model to N sampled from the manifold's embedding
        """
        embedded_points_pca = [
            Point(self.manifold.embedding(p), self.manifold.embedding)
            for p in self.manifold.sample(n=self.pca_sample_points, full=True)
        ]
        embedded = np.vstack([p.coordinates for p in embedded_points_pca])

        self.pca = PCA(n_components=3).fit(embedded)
        self.embedded_lowd = self.pca.transform(embedded)

    def visualize_manifold(self):
        for point in self.manifold.points:
            lowd_coords = self.pca.transform(
                np.array(point.embedded).reshape(1, -1)
            )
            self.actors.append(Spheres(lowd_coords, r=0.15, c=blue,))

        self.actors.append(
            recoSurface(self.embedded_lowd, dims=(20, 20, 20), radius=0.5)
            .c(grey)
            .clean()
            .alpha(0.8)
        )

    def visualize_tangent_vectors(self, scale=1, x_range=0.05):
        if not isinstance(x_range, list):
            x_range = [x_range] * self.manifold.d

        for point in self.manifold.points:
            weights = self.manifold.vectors_field(point)
            vectors = []
            for n, fn in enumerate(point.base_functions):
                fn.embedd(x_range=x_range[fn.dim_idx])
                vectors.append(fn.tangent_vector * weights[n])

                low_d = self.pca.transform(fn.embedded)
                self.actors.append(Tube(low_d, r=0.02, c=grey_dark,))

            point_lowd = self.pca.transform(
                np.array(point.embedded).reshape(1, -1)
            ).T
            vector = np.sum(np.vstack(vectors), 0)
            vec_lowd = self.pca.transform(vector.reshape(-1, 1).T)[0] * scale
            pts = np.vstack(
                [
                    [point_lowd[0][0], (point_lowd[0] + vec_lowd[0])[0]],
                    [point_lowd[1][0], (point_lowd[1] + vec_lowd[1])[0]],
                    [point_lowd[2][0], (point_lowd[2] + vec_lowd[2])[0]],
                ]
            ).T
            self.actors.append(Tube(pts, r=0.03, c=green,))

    def visualize_rnn_traces(self):
        for trace in self.rnn.traces:
            lowd = self.pca.transform(trace.trace)
            self.actors.append(Tube(lowd, c=salmon, r=0.02,))

    def show(self, scale=1, x_range=0.05):
        self.visualize_manifold()
        self.visualize_tangent_vectors(scale=scale, x_range=x_range)

        if self.rnn is not None:
            self.visualize_rnn_traces()

        for actor in self.actors:
            actor.lighting("plastic")

        camera = dict(
            pos=[-0.185, -11.551, 8.326],
            focalPoint=[0.115, -0.647, 0.251],
            viewup=[-0.039, 0.595, 0.802],
            distance=14,
            # clippingRange = [7.661, 22.088],
        )

        show(
            *self.actors,
            size="full",
            title=self.manifold.name,
            axes=1,
            camera=camera,
        )
