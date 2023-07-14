import open3d as o3d
import numpy as np
from lidarnvs.lidarnvs_meshing import LidarNVSMeshing

import torch
import nksr


class LidarNVSNksr(LidarNVSMeshing):
    def __init__(self, ckpt_path=None):
        super(LidarNVSNksr, self).__init__(ckpt_path=ckpt_path)

        # To be filled in the fit() method.
        self.points = None
        self.point_intensities = None
        self.pcd = None
        self.kdtree = None
        self.mesh = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nksr_reconstructor = nksr.Reconstructor(self.device)

        # self.meshing_func shall be pre-filled with functools.partial.
        self.meshing_func = self._run_nksr

    def _run_nksr(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> o3d.geometry.TriangleMesh:
        print("Start _run_nksr()")

        pcd.estimate_normals()

        input_xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)
        input_normal = torch.from_numpy(np.asarray(pcd.normals)).float().to(self.device)

        field = self.nksr_reconstructor.reconstruct(
            input_xyz, input_normal, detail_level=0.5
        )
        mesh = field.extract_dual_mesh(mise_iter=1)

        vertices = mesh.v.cpu().numpy()
        triangles = mesh.f.cpu().numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        return mesh
