from sklearn.decomposition import PCA
import numpy as np
from vedo import (
    Tube,
    recoSurface,
    Line,
    Sphere,
    Plotter,
    Cylinder,
    Torus,
    Cone,
)

from myterial import (
    grey,
    salmon,
    green,
    black,
    blue_light,
    indigo,
    deep_purple,
)

from manifold.tangent_vector import get_tangent_vector
from manifold._visualize import blue_dark, make_palette
from manifold.maths import unit_vector

class Visualizer:
    actors = []

    def __init__(self, manifold, rnn=None, pca_sample_points=64):
        self.manifold = manifold
        self.rnn = rnn

        if self.manifold.n > 3:
            self.pca_sample_points = pca_sample_points
            self.fit_pca()

        self.make_plotter()

    def make_plotter(self):
        if self.manifold.n > 3:
            camera = dict(
                pos=[-0.185, -11.551, 8.326],
                focalPoint=[0.115, -0.647, 0.251],
                viewup=[-0.039, 0.595, 0.802],
                distance=14,
                clippingRange=[2.661, 22.088],
            )
        else:
            camera = dict(
                pos=(-5.16, 3.38, 1.29),
                focalPoint=(4.83e-3, -0.299, -0.275),
                viewup=(0.183, -0.155, 0.971),
                distance=6.53,
                clippingRange=(3.11, 10.4),
            )

        self.plotter = Plotter(size="full", axes=1)

        self.plotter.camera.SetPosition(camera["pos"])
        self.plotter.camera.SetFocalPoint(camera["focalPoint"])
        self.plotter.camera.SetViewUp(camera["viewup"])
        self.plotter.camera.SetDistance(camera["distance"])
        self.plotter.camera.SetClippingRange(camera["clippingRange"])

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

    def _add_silhouette(self, mesh, lw=2):
        self.actors.append(mesh.silhouette().lw(lw).color(black))

    def _render_cylinder(self, pts, color, r=0.02):
        mesh = Cylinder(pts, r=r, c=color,)
        self.actors.append(mesh)
        self._add_silhouette(mesh)

    def _scatter_point(self, point):
        if self.manifold.n > 3:
            coordinates = self.pca.transform(
                np.array(point.embedded).reshape(1, -1)
            )[0]
        else:
            coordinates = np.array(point.embedded)

        mesh = Sphere(
            coordinates, r=0.05 if self.manifold.d > 1 else 0.05, c=blue_dark,
        )
        self.actors.append(mesh)
        self._add_silhouette(mesh)

    def _reconstruct_surface(self, coordinates):
        # plot points
        self.actors.append(
            recoSurface(coordinates, dims=(50, 50, 50), radius=0.15)
            .c(grey)
            .wireframe()
            .lw(1)
            .clean()
        )

    def _draw_manifold(self):
        if self.manifold.n == 3:
            if self.manifold.d == 1:
                self.actors.append(
                    Line(self.manifold.embedded, lw=12, c=grey,)
                )
            else:
                if self.manifold.name == "S2":
                    # plot a sphere
                    self.actors.append(Sphere(r=0.75, c=grey).wireframe())

                elif self.manifold.name == "Cy":
                    # plot a cylinder
                    self.actors.append(
                        Cone(pos=(0, 0, 1.5), r=1, axis=(0, 0, -1), c=grey)
                        .wireframe()
                        .lw(2)
                    )

                elif self.manifold.name == "T2":
                    # plot a torus
                    self.actors.append(
                        Torus(r=0.5, thickness=0.25, c="grey", res=20,)
                        .wireframe()
                        .lw(1)
                    )

                else:
                    # plot points
                    self._reconstruct_surface(self.manifold.embedded)
        else:
            if self.manifold.d > 1:
                self._reconstruct_surface(self.embedded_lowd)
            else:
                self.actors.append(Line(self.embedded_lowd, c=grey, lw=12))

    def visualize_manifold(self):
        for point in self.manifold.points:
            self._scatter_point(point)
        self._draw_manifold()

    def visualize_tangent_vectors(self, scale=1, x_range=0.05):
        if not isinstance(x_range, list):
            x_range = [x_range] * self.manifold.d

        for point in self.manifold.points:
            # draw base functions
            for fn in point.base_functions:
                fn.embedd(x_range=x_range[fn.dim_idx])
                # if self.manifold.n == 3:
                #     coordinates = fn.embedded
                # else:
                #     coordinates = self.pca.transform(fn.embedded)

                # mesh = Tube(coordinates, r=0.02, c=grey_dark,)
                # self.actors.append(mesh)
                # self._add_silhouette(mesh, lw=100)

            # get tangent vector as sum of basis
            vector = get_tangent_vector(
                point, self.manifold.vectors_field, debug=True
            ) * scale + np.array(point.embedded)

            # apply PCA and render
            if self.manifold.n > 3:
                point_lowd = self.pca.transform(
                    np.array(point.embedded).reshape(1, -1)
                ).ravel()
                vec_lowd = self.pca.transform(vector.reshape(1, -1)).ravel()

                self._render_cylinder([point_lowd, vec_lowd], green, r=0.02)
            else:
                self._render_cylinder([point.embedded, vector], green, r=0.02)

    def visualize_rnn_inputs(self, scale=1, rnn_inputs=None):
        """
            Plots the basis vectors of the RNN's inputs vectors space
        """
        if self.rnn.B is None:
            return
        if self.manifold.n > 3:
            raise NotImplementedError("Reduce dim on point and vec")

        # visualize inputs basis vector
        # colors = make_palette(blue_light, indigo, self.rnn.n_inputs)
        colors = ['r', 'r', 'r', 'r', 'g', 'g', 'g', 'g']
        for base in self.rnn.inputs_basis:
            for point in self.manifold.points:
                self._render_cylinder(
                    [
                        point.embedded,
                        point.embedded + base.projected * scale * 0.5,
                    ],
                    colors[base.idx],
                    r=0.02,
                )

        if rnn_inputs is not None:
            if len(rnn_inputs.shape) == 1:
                rnn_inputs = rnn_inputs.reshae(1, -1)
            
            for idx in np.arange(rnn_inputs.shape[0]):
                try:
                    vec = unit_vector(self.rnn.B.T @ rnn_inputs[idx])
                except ValueError:
                    raise ValueError(
                        f"Failed to compute RNN input vec Bu - B shape: {self.rnn.B.shape} - u shape {rnn_inputs.shape}"
                    )

                for point in self.manifold.points:
                    self._render_cylinder(
                        [point.embedded, point.embedded + vec], deep_purple, r=0.02
                    )


    def visualize_rnn_traces(self):
        for trace in self.rnn.traces:
            if self.manifold.n > 3:
                coordinates = self.pca.transform(trace.trace)
            else:
                coordinates = trace.trace
            self.actors.append(
                Tube(
                    coordinates,
                    c=salmon,
                    r=0.02 if self.manifold.d == 1 else 0.015,
                )
            )

    def show(self, scale=1, x_range=0.05, rnn_inputs=None):
        self.visualize_manifold()
        self.visualize_tangent_vectors(scale=scale, x_range=x_range)

        if self.rnn is not None:
            self.visualize_rnn_traces()
            self.visualize_rnn_inputs(scale=scale, rnn_inputs=rnn_inputs)

        for actor in self.actors:
            actor.lighting("off")

        self.plotter.show(*self.actors)
