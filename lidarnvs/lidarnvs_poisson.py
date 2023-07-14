import time

import open3d as o3d
import numpy as np
from lidarnvs.lidarnvs_meshing import LidarNVSMeshing
import functools


class LidarNVSPoisson(LidarNVSMeshing):
    @staticmethod
    def _run_poisson(
        pcd: o3d.geometry.PointCloud,
        depth: int,
        min_density: int,
    ) -> o3d.geometry.TriangleMesh:
        print("Start _run_poisson()")
        s_time = time.time()
        # Run.
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
        )
        # Filter by density.
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        # All-black colors are generated, but we don't need them.
        mesh.vertex_colors = o3d.utility.Vector3dVector([])
        print(f"_run_poisson() time: {time.time() - s_time:.3f} secs")
        return mesh

    def __init__(
        self,
        poisson_depth=10,
        poisson_min_density=0.6,
        intensity_interpolate_k=5,
        ckpt_path=None,
    ):
        super(LidarNVSPoisson, self).__init__(ckpt_path=ckpt_path)

        self.poisson_depth = poisson_depth
        self.poisson_min_density = poisson_min_density
        self.intensity_interpolate_k = intensity_interpolate_k

        # To be filled in the fit() method.
        self.points = None
        self.point_intensities = None
        self.pcd = None
        self.kdtree = None
        self.mesh = None

        # self.meshing_func shall be pre-filled with functools.partial.
        self.meshing_func = functools.partial(
            LidarNVSPoisson._run_poisson,
            depth=self.poisson_depth,
            min_density=self.poisson_min_density,
        )
