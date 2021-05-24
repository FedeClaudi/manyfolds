from sklearn.decomposition import PCA
import numpy as np

from myterial import grey, blue, salmon, green, grey_dark

from manifold.topology import Point
from manifold.visualize import make_3D_ax


class Visualizer:
    """
        Class to facilitate the visualization of 
        data from embeddings in N >> 3 using 
        dimensionality reduction techniques.
    """

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

    def visualize_manifold(self, ax_lims=True):
        ax = make_3D_ax(nolim=ax_lims)
        ax.set(xlabel="PC 1", ylabel="PC 2", zlabel="PC 3")

        for point in self.manifold.points:
            lowd_coords = self.pca.transform(
                np.array(point.embedded).reshape(1, -1)
            )
            ax.scatter(
                *lowd_coords.T,
                s=100,
                color=blue,
                zorder=2,
                edgecolors=grey,
                lw=0.5,
                alpha=0.5,
            )

        if self.manifold.d == 1:
            ax.plot(
                self.embedded_lowd[:, 0],
                self.embedded_lowd[:, 1],
                self.embedded_lowd[:, 2],
                lw=1.5,
                color=grey,
                zorder=-1,
            )
        else:
            for n in range(self.pca_sample_points):
                idxs = [
                    k + self.pca_sample_points * n
                    for k in range(self.pca_sample_points)
                    if k + self.pca_sample_points * n
                    < self.embedded_lowd.shape[0]
                ]
                ax.plot(
                    self.embedded_lowd[idxs, 0],
                    self.embedded_lowd[idxs, 1],
                    self.embedded_lowd[idxs, 2],
                    lw=1.5,
                    color=grey,
                    zorder=-1,
                )

        return ax

    def visualize_tangent_vectors(self, ax, scale=1, x_range=0.05):
        if not isinstance(x_range, list):
            x_range = [x_range] * self.manifold.d

        for point in self.manifold.points:
            weights = self.manifold.vectors_field(point)
            vectors = []
            for n, fn in enumerate(point.base_functions):
                fn.embedd(x_range=x_range[fn.dim_idx])
                vectors.append(fn.tangent_vector * weights[n])

                low_d = self.pca.transform(fn.embedded)
                ax.plot(
                    low_d[:, 0],
                    low_d[:, 1],
                    low_d[:, 2],
                    lw=5,
                    color=grey_dark,
                )

            point_lowd = self.pca.transform(
                np.array(point.embedded).reshape(1, -1)
            ).T
            vector = np.sum(np.vstack(vectors), 0)
            vec_lowd = self.pca.transform(vector.reshape(-1, 1).T)[0] * scale
            ax.plot(
                [point_lowd[0][0], (point_lowd[0] + vec_lowd[0])[0]],
                [point_lowd[1][0], (point_lowd[1] + vec_lowd[1])[0]],
                [point_lowd[2][0], (point_lowd[2] + vec_lowd[2])[0]],
                lw=4,
                color=green,
            )

    def visualize_rnn_traces(self, ax):
        for trace in self.rnn.traces:
            lowd = self.pca.transform(trace.trace)
            ax.plot(
                lowd[:, 0], lowd[:, 1], lowd[:, 2], c=salmon, lw=4,
            )

    def show(self, ax_lims=True, scale=1, x_range=0.05):
        ax = self.visualize_manifold(ax_lims=ax_lims)
        self.visualize_tangent_vectors(ax, scale=scale, x_range=x_range)

        if self.rnn is not None:
            self.visualize_rnn_traces(ax)

        return ax
