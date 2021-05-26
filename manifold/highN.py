from sklearn.decomposition import PCA
import numpy as np
from vedo import show, Spheres, Tube, recoSurface, Line

from myterial import grey, blue, salmon, green, grey_dark

from manifold.tangent_vector import get_tangent_vector


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
        embedded_points_pca = self.manifold.sample(
            n=self.pca_sample_points, full=True, fill=True
        )
        embedded = np.vstack([p.embedded for p in embedded_points_pca])

        self.pca = PCA(n_components=3).fit(embedded)
        self.embedded_lowd = self.pca.transform(embedded)

    def visualize_manifold(self):
        for point in self.manifold.points:
            lowd_coords = self.pca.transform(
                np.array(point.embedded).reshape(1, -1)
            )
            self.actors.append(
                Spheres(
                    lowd_coords,
                    r=0.05 if self.manifold.d > 1 else 0.05,
                    c=blue,
                )
            )

        if self.manifold.d > 1:
            self.actors.append(
                recoSurface(self.embedded_lowd, dims=(50, 50, 50), radius=0.1)
                .c(grey)
                .clean()
                .alpha(0.8)
                .wireframe()
                .lw(1)
            )
        else:
            self.actors.append(Line(self.embedded_lowd, c=grey, lw=4))

    def visualize_tangent_vectors(self, scale=1, x_range=0.05):
        if not isinstance(x_range, list):
            x_range = [x_range] * self.manifold.d

        for point in self.manifold.points:
            # draw base functions
            for fn in point.base_functions:
                fn.embedd(x_range=x_range[fn.dim_idx])
                low_d = self.pca.transform(fn.embedded)
                self.actors.append(Tube(low_d, r=0.02, c=grey_dark,))

            # get tangent vector as sum of basis
            vector = get_tangent_vector(
                point, self.manifold.vectors_field, debug=True
            ) * scale + np.array(point.embedded)

            # apply PCA and render
            point_lowd = self.pca.transform(
                np.array(point.embedded).reshape(1, -1)
            ).ravel()
            vec_lowd = self.pca.transform(vector.reshape(1, -1)).ravel()

            self.actors.append(Tube([point_lowd, vec_lowd], r=0.03, c=green,))

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
            clippingRange=[2.661, 22.088],
        )

        show(
            *self.actors,
            size="full",
            title=self.manifold.name,
            axes=1,
            camera=camera,
        )
