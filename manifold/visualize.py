from sklearn.decomposition import PCA
import numpy as np
from loguru import logger
from vedo import (
    Tube,
    recoSurface,
    Sphere,
    Plotter,
    Cylinder,
    Torus,
)

from myterial import (
    salmon,
    green,
    black,
    amber_dark,
    pink_dark,
)

from manifold.tangent_vector import (
    get_tangent_vector,
    get_basis_tangent_vector,
)
from manifold._visualize import make_palette

# settings
point_size = 0.05
reco_surface_radius = 0.3
rnn_trace_radius = 0.03
rnn_inputs_radius = 0.02
tangent_vector_radius = 0.02
manifold_1d_lw = 12
manifold_1d_r = 0.015


class Visualizer:
    actors = []

    def __init__(
        self,
        manifold,
        rnn=None,
        pca_sample_points=64,
        point_color=None,
        camera=None,
        manifold_alpha=0.7,
        axes=None,
        wireframe=False,
        manifold_color=None,
        mark_rnn_endpoint=False,
    ):
        self.manifold = manifold
        self.rnn = rnn
        self.wireframe = wireframe
        self.manifold_color = manifold_color or "#b8b6d1"
        self.point_color = point_color or "#3838BA"
        self.manifold_alpha = manifold_alpha
        self.mark_rnn_endpoint = mark_rnn_endpoint

        if self.manifold.n > 3:
            self.pca_sample_points = pca_sample_points
            self.fit_pca()

        self.make_plotter(axes=axes, camera=camera)

    def make_plotter(self, axes=None, camera=None):
        if camera is None:
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

        if axes is None:
            axes = 1
        self.plotter = Plotter(size="full", axes=axes)

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

        if self.manifold.shift_applied:
            embedded += self.manifold.points_shift

        self.pca = PCA(n_components=3).fit(embedded)
        self.embedded_lowd = self.pca.transform(embedded)

    def _add_silhouette(self, mesh, lw=2):
        self.actors.append(mesh.silhouette().lw(lw).color(black))

    def _render_cylinder(self, pts, color, r=0.02, alpha=1):
        mesh = Cylinder(pts, r=r, c=color, alpha=alpha)
        self.actors.append(mesh)
        self._add_silhouette(mesh)

    def _scatter_point(self, point):
        if self.manifold.n > 3:
            coordinates = self.pca.transform(point.embedded.reshape(1, -1))[0]
        else:
            coordinates = point.embedded

        mesh = Sphere(coordinates, r=point_size, c=self.point_color,)
        self.actors.append(mesh)
        self._add_silhouette(mesh)

    def _reconstruct_surface(self, coordinates):
        # plot points
        manifold = (
            recoSurface(
                coordinates, dims=(50, 50, 50), radius=reco_surface_radius
            )
            .c(self.manifold_color)
            .clean()
        )

        if self.wireframe:
            manifold = manifold.wireframe().lw(2)
        else:
            self._add_silhouette(manifold)
        self.actors.append(manifold)
        return manifold

    def _draw_manifold(self):
        if self.manifold.n == 3:
            if self.manifold.d == 1:
                manifold = Tube(
                    self.manifold.embedded,
                    r=manifold_1d_r,
                    c=self.manifold_color,
                )
                self.actors.append(manifold)
            else:
                if self.manifold.name in ("S^2", "Cy", "T^2"):
                    if self.manifold.name == "S^2":
                        # plot a sphere
                        manifold = Sphere(r=0.99, c=self.manifold_color)

                    elif self.manifold.name == "Cy":
                        # plot a cylinder
                        # manifold = Cone(pos=(0, 0, 1.5), r=1, axis=(0, 0, -1), c=self.manifold_color)
                        manifold = Cylinder(
                            pos=(0, 0, 0.5),
                            height=1.5,
                            r=0.5,
                            axis=(0, 0, -1),
                            c=self.manifold_color,
                        )

                    elif self.manifold.name == "T^2":
                        # plot a torus
                        manifold = Torus(
                            r=0.73,
                            thickness=0.24,
                            c=self.manifold_color,
                            res=20,
                        )

                    if self.wireframe:
                        manifold = manifold.wireframe().lw(1.5)

                    self._add_silhouette(manifold, lw=2)
                    self.actors.append(manifold)

                else:
                    # plot points
                    manifold = self._reconstruct_surface(
                        self.manifold.embedded
                    )
        else:
            if self.manifold.d > 1:
                manifold = self._reconstruct_surface(self.embedded_lowd)
            else:
                manifold = Tube(
                    self.embedded_lowd, c=self.manifold_color, r=manifold_1d_r
                )
                self.actors.append(manifold)

        self.manifold_actor = manifold.alpha(self.manifold_alpha)

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

                # mesh = Tube(coordinates, r=0.02, c=(1 - fn.dim_idx * .2) * np.array([.3, .3, .3]),)
                # self.actors.append(mesh)
                # self._add_silhouette(mesh, lw=100)

            # get tangent vector as sum of basis
            vector = (
                get_tangent_vector(
                    point, self.manifold.vectors_field, debug=False
                )
                * scale
                + point.embedded
            )

            # apply PCA and render
            if self.manifold.n > 3:
                point_lowd = self.pca.transform(
                    point.embedded.reshape(1, -1)
                ).ravel()
                try:
                    vec_lowd = self.pca.transform(
                        vector.reshape(1, -1)
                    ).ravel()
                except ValueError as e:
                    logger.warning(
                        f"Could not visualize PCA transformed tangent vector: {e}"
                    )
                else:
                    self._render_cylinder(
                        [point_lowd, vec_lowd], green, r=tangent_vector_radius
                    )
            else:
                self._render_cylinder(
                    [point.embedded, vector], green, r=tangent_vector_radius
                )

    def visualize_basis_vectors_at_point(
        self, point, color="k", r=0.03, scale=0.4
    ):
        pt = point.embedded
        for fn in point.base_functions:
            fn.embedd()
            vec = get_basis_tangent_vector(point, fn) * scale
            if self.manifold.n > 3:
                pt = point.embedded
                vec = self.pca.transform((pt + vec).reshape(1, -1))[0]
                pt = self.pca.transform((pt).reshape(1, -1))[0]
            else:
                vec = pt + vec
            self._render_cylinder([pt, vec], color, r=r)

    def visualize_rnn_inputs(self, scale=1, rnn_inputs=None):
        """
            Plots the basis vectors of the RNN's inputs vectors space
        """
        if self.rnn.B is None:
            return

        # visualize inputs basis vector
        colors = make_palette(amber_dark, pink_dark, self.rnn.n_inputs)
        for base in self.rnn.inputs_basis:
            for point in self.manifold.points:
                if self.manifold.n == 3:
                    pt = point.embedded
                    vec = point.embedded + base.projected * scale
                else:
                    pt = self.pca.transform(point.embedded.reshape(1, -1))[0]
                    vec = point.embedded + base.projected * scale
                    vec = self.pca.transform(vec.reshape(1, -1))[0]
                self._render_cylinder(
                    [pt, vec], colors[base.idx], r=rnn_inputs_radius,
                )
                # break

    def visualize_rnn_traces(self):
        for trace in self.rnn.traces:
            if self.manifold.n > 3:
                try:
                    coordinates = self.pca.transform(trace.trace)
                except ValueError as e:
                    logger.warning(f"Could not PCA transform RNN trace: {e}")
                    continue
            else:
                coordinates = trace.trace
            self.actors.append(
                Tube(coordinates, c=salmon, r=rnn_trace_radius,)
            )

            if self.mark_rnn_endpoint:
                point = Sphere(
                    coordinates[-1, :],
                    c=[0.3, 0.3, 0.3],
                    r=point_size + 0.2 * point_size,
                )
                self._add_silhouette(point)
                self.actors.append(point)

    def show(
        self,
        scale=1,
        x_range=0.05,
        rnn_inputs=None,
        show_tangents=True,
        show_rnn_inputs_vectors=True,
        show_manifold=True,
        **kwargs,
    ):
        if show_manifold:
            self.visualize_manifold()
        else:
            for point in self.manifold.points:
                self._scatter_point(point)

        if show_tangents:
            self.visualize_tangent_vectors(scale=scale, x_range=x_range)

        if self.rnn is not None:
            self.visualize_rnn_traces()

            if show_rnn_inputs_vectors:
                self.visualize_rnn_inputs(scale=scale, rnn_inputs=rnn_inputs)

        for actor in self.actors:
            actor.lighting("off")

        self.plotter.show(*self.actors, **kwargs)
