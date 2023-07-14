import camtools as ct
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lidarnerf.convert import (
    lidar_to_pano_with_intensities,
    pano_to_lidar_with_intensities,
)
from lidarnerf.dataset.base_dataset import get_lidar_rays
from lidarnvs.lidarnvs_base import LidarNVSBase
from lidarnvs.loader import extract_dataset_frame
from lidarnvs.unet import UNet


class LidarNVSMeshing(LidarNVSBase):
    """
    Liar novel-view synthesis with meshing and ray casting.

    This is intended to be a base class, where the children class can use
    different meshing methods.
    """

    def __init__(self, ckpt_path=None):
        self.ckpt_path = ckpt_path

        # Network for predicting ray-drop.
        if ckpt_path is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = UNet(n_channels=10, n_classes=1, bilinear=False)
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.to(device=self.device)

            state_dict = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Checkpoint loaded from {self.ckpt_path}")

        # To be filled in the fit() method.
        self.points = None
        self.point_intensities = None
        self.pcd = None
        self.kdtree = None
        self.mesh = None

        # To be overwritten by the child class.
        # - meshing_func: o3d.geometry.PointCloud -> o3d.geometry.TriangleMesh
        # - the meshing_func shall already be populated with hyper-parameters
        self.meshing_func = None

    def fit(self, dataset) -> None:
        """
        Fit the model to the given train dataset.

        Args:
            dataset: A NeRFDataset object.
        """
        # Extract all points, in world coordinates.
        num_frames = len(dataset)
        all_points = []
        all_point_intensities = []
        for frame_idx in tqdm(range(num_frames), "Extract train frames"):
            frame_dict = extract_dataset_frame(dataset, frame_idx)
            all_points.append(frame_dict["points"])
            all_point_intensities.append(frame_dict["point_intensities"])
        all_points = np.vstack(all_points)

        all_point_intensities = np.hstack(all_point_intensities)
        assert len(all_points) == len(all_point_intensities)

        # Build Open3D pcd.
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(all_points)
        colors = ct.colormap.query(all_point_intensities)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.pcd.estimate_normals()

        # Save points and intensities for interpolation.
        self.points = all_points
        self.point_intensities = all_point_intensities

        # Run Poisson recon.
        self.mesh = self.meshing_func(self.pcd)
        self.mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([self.mesh])

        # Build kdtree for kNN search.
        self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)

        # Build scene for ray casting.
        self.raycasting_scene = o3d.t.geometry.RaycastingScene()
        self.raycasting_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        )

    def predict_frame(
        self,
        lidar_K: np.ndarray,  # (2, )
        lidar_pose: np.ndarray,  # (4, 4)
        lidar_H: int,
        lidar_W: int,
    ) -> dict:
        """
        Predict (synthesis) the point cloud from the given lidar parameters.
        All necessary information parameters to model a lidar are given.

        Args:
            lidar_K: (2, ), float32
            lidar_pose: (4, 4), float32
            lidar_H: int
            lidar_W: int

        Return:
            predict_dict: dict
            - ["local_points"]: (N, 3), float32
            - ["points"]      : (N, 3), float32
            - ["pano"]        : (H, W), float32
            - ["intensities"] : (H, W), float32
        """
        # In world and local coordinates.
        hit_dict = self.intersect_lidar(lidar_K, lidar_pose, lidar_H, lidar_W)
        points = hit_dict["points"][hit_dict["masks"]]
        local_points = ct.project.homo_project(
            points,
            ct.convert.pose_to_T(lidar_pose),
        )

        # Point intensities in world/local coordinates.
        point_intensities = []
        for point in points:
            # ks, indices, distances2
            _, indices, _ = self.kdtree.search_knn_vector_3d(
                point, self.intensity_interpolate_k
            )
            point_intensities.append(np.mean(self.point_intensities[indices]))
        point_intensities = np.array(point_intensities)
        local_point_intensities = point_intensities

        # Pano intensities.
        local_points_with_intensities = np.concatenate(
            [local_points, local_point_intensities.reshape((-1, 1))], axis=1
        )
        pano, intensities = lidar_to_pano_with_intensities(
            local_points_with_intensities=local_points_with_intensities,
            lidar_H=lidar_H,
            lidar_W=lidar_W,
            lidar_K=lidar_K,
        )

        predict_dict = {
            # Frame properties.
            "pano": pano,
            "intensities": intensities,
            # Global properties.
            "points": points,
            "point_intensities": point_intensities,
            # Local properties.
            "local_points": local_points,
            "local_point_intensities": local_point_intensities,
            # Hit properties: unfiltered results from ray casting.
            "hit_dict": hit_dict,
        }
        return predict_dict

    @torch.inference_mode()
    def predict_frame_with_raydrop(
        self,
        lidar_K: np.ndarray,  # (2, )
        lidar_pose: np.ndarray,  # (4, 4)
        lidar_H: int,
        lidar_W: int,
    ) -> dict:
        """
        TODO: I know this is ugly. This is the manual combination of:
        - generate_raydrop_data()
        - RaydropDataset::collate_fn()
        """
        nvs_frame = self.predict_frame(
            lidar_K=lidar_K,
            lidar_pose=lidar_pose,
            lidar_H=lidar_H,
            lidar_W=lidar_W,
        )

        # Compute incidence angle cosine.
        # TODO: make get rays a function.
        ray_dict = get_lidar_rays(
            poses=torch.tensor(np.array([lidar_pose])),
            intrinsics=torch.tensor(lidar_K),
            H=torch.tensor(lidar_H),
            W=torch.tensor(lidar_W),
        )

        # generate_raydrop_data() ############################################
        rays_o = ray_dict["rays_o"].squeeze().numpy()
        rays_d = ray_dict["rays_d"].squeeze().numpy()
        hit_normals = nvs_frame["hit_dict"]["normals"]
        hit_incidences = np.abs(np.sum(rays_d * hit_normals, axis=-1))

        # Reshape.
        hit_masks = nvs_frame["hit_dict"]["masks"]
        hit_masks = hit_masks.reshape((lidar_H, lidar_W))
        hit_depths = nvs_frame["hit_dict"]["depths"]
        hit_depths[hit_depths == np.inf] = 0
        hit_depths = hit_depths.reshape((lidar_H, lidar_W))
        hit_normals = hit_normals.reshape((lidar_H, lidar_W, 3))
        hit_incidences = hit_incidences.reshape((lidar_H, lidar_W))
        intensities = nvs_frame["intensities"]
        intensities = intensities.reshape((lidar_H, lidar_W))
        rays_o = rays_o.reshape((lidar_H, lidar_W, 3))
        rays_d = rays_d.reshape((lidar_H, lidar_W, 3))

        # Cast
        hit_masks = torch.tensor(hit_masks.astype(np.float32))
        hit_depths = torch.tensor(hit_depths.astype(np.float32))
        hit_normals = torch.tensor(hit_normals.astype(np.float32))
        hit_incidences = torch.tensor(hit_incidences.astype(np.float32))
        intensities = torch.tensor(intensities.astype(np.float32))
        rays_o = torch.tensor(rays_o.astype(np.float32))
        rays_d = torch.tensor(rays_d.astype(np.float32))

        # Add batch dimension 1 to the front
        hit_masks = hit_masks.unsqueeze(0)
        hit_depths = hit_depths.unsqueeze(0)
        hit_normals = hit_normals.unsqueeze(0)
        hit_incidences = hit_incidences.unsqueeze(0)
        intensities = intensities.unsqueeze(0)
        rays_o = rays_o.unsqueeze(0)
        rays_d = rays_d.unsqueeze(0)
        ######################################################################

        # RaydropDataset::collate_fn() #######################################
        # (N, H, W, C)
        images = torch.cat(
            [
                hit_masks[..., None].to(self.device),
                hit_depths[..., None].to(self.device),
                hit_normals.to(self.device),
                hit_incidences[..., None].to(self.device),
                intensities[..., None].to(self.device),
                rays_d.to(self.device),
            ],
            dim=3,
        )
        # (N, C, H, W)
        images = images.permute(0, 3, 1, 2)
        ######################################################################

        # Predict raydrop mask.
        pd_raydrop_masks = self.model(images)
        pd_raydrop_masks = (F.sigmoid(pd_raydrop_masks) > 0.5).float()
        pd_raydrop_masks = pd_raydrop_masks.squeeze().cpu().numpy()
        if False:
            plt.imshow(pd_raydrop_masks)
            plt.show()

        # Update predict_dict
        # Frame properties.
        pano = nvs_frame["pano"] * pd_raydrop_masks
        intensities = nvs_frame["intensities"] * pd_raydrop_masks
        # Local properties.
        local_points_with_intensities = pano_to_lidar_with_intensities(
            pano=pano,
            intensities=intensities,
            lidar_K=lidar_K,
        )
        local_points = local_points_with_intensities[:, :3]
        local_point_intensities = local_points_with_intensities[:, 3]
        # Global properties.
        points = ct.project.homo_project(local_points, lidar_pose)
        point_intensities = local_point_intensities

        predict_dict = {
            # Frame properties.
            "pano": pano,
            "intensities": intensities,
            # Global properties.
            "points": points,
            "point_intensities": point_intensities,
            # Local properties.
            "local_points": local_points,
            "local_point_intensities": local_point_intensities,
            # Hit properties: unfiltered results from ray casting.
            "hit_dict": nvs_frame["hit_dict"],
        }

        return predict_dict

    def intersect_rays(self, rays):
        """
        Compute ray-mesh intersect and return the hit_dict.
        The hit_dict will NOT be filtered, but the masks will be provided.

        Args:
            mesh: o3d.geometry.TriangleMesh
            rays: (N, 6), float32 rays, where rays[:, :3] is the origin and
                rays[:, 3:] is the direction. The directions do not need to be
                normalized.

        Return:
            hit_dict
            - ["masks"]  : (N, ) , boolean mask of ray hit.
            - ["depths"]  : (N, ) , depth in world-scale.
            - ["points"] : (N, 3), coordinates of the intersection points.
            - ["normals"]: (N, 3), normals of the hit triangles.
        """
        # Sanity checks.
        if not isinstance(rays, np.ndarray):
            raise TypeError("rays must be a numpy array.")
        if rays.ndim != 2 or rays.shape[1] != 6:
            raise ValueError("rays must be a (N, 6) array.")

        # Run ray cast.
        ray_cast_results = self.raycasting_scene.cast_rays(o3c.Tensor(rays))
        normals = ray_cast_results["primitive_normals"].numpy()
        depths = ray_cast_results["t_hit"].numpy()
        masks = depths != np.inf
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:]
        rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)
        points = rays_o + rays_d * depths[:, None]

        hit_dict = {
            "masks": masks,
            "depths": depths,
            "points": points,
            "normals": normals,
        }

        return hit_dict

    def intersect_lidar(
        self,
        lidar_K: np.ndarray,  # (2, )
        lidar_pose: np.ndarray,  # (4, 4)
        lidar_H: int,
        lidar_W: int,
    ):
        ray_dict = get_lidar_rays(
            poses=torch.tensor(np.array([lidar_pose])),
            intrinsics=torch.tensor(lidar_K),
            H=torch.tensor(lidar_H),
            W=torch.tensor(lidar_W),
        )
        rays_o = ray_dict["rays_o"].squeeze().numpy()
        rays_d = ray_dict["rays_d"].squeeze().numpy()
        rays = np.concatenate([rays_o, rays_d], axis=-1)
        hit_dict = self.intersect_rays(rays)
        return hit_dict


def generate_raydrop_data_meshing(dataset, nvs: LidarNVSMeshing) -> dict:
    """
    Prepare dataset for learning ray drop.
    The frames are NOT loaded by our dataset, but GENERATED.

    Return:
        raydrop_data = [
            {
                "hit_masks"      : (H, W)    # Ray cast hit mask
                "hit_depths"     : (H, W)    # Hit intersection point depths
                "hit_normals"    : (H, W, 3) # Intersection point normal
                "hit_incidences" : (H, W)    # |cos(normal, ray_d)|
                "intensities"    : (H, W)    # Predicted intensities
                "rays_o"         : (H, W, 3) # Lidar ray origin
                "rays_d"         : (H, W, 3) # Lidar ray direction
                "raydrop_masks"  : (H, W)    # Ray drop mask, 1 is valid
            },
            ...
        ]
    """
    raydrop_data = []
    for frame_idx in tqdm(range(len(dataset)), desc="Prepare raydrop dataset"):
        gt_frame = extract_dataset_frame(dataset, frame_idx=frame_idx)
        nvs_frame = nvs.predict_frame(
            lidar_K=gt_frame["lidar_K"],
            lidar_pose=gt_frame["lidar_pose"],
            lidar_H=gt_frame["lidar_H"],
            lidar_W=gt_frame["lidar_W"],
        )

        # The target.
        raydrop_masks = gt_frame["pano"] != 0

        # Compute incidence angle cosine.
        rays_o = gt_frame["rays"][:, :3]
        rays_d = gt_frame["rays"][:, 3:]
        hit_normals = nvs_frame["hit_dict"]["normals"]
        hit_incidences = np.abs(np.sum(rays_d * hit_normals, axis=-1))

        # Pre-processing.
        # TODO: move the reshape to upper-level
        lidar_H, lidar_W = gt_frame["lidar_H"], gt_frame["lidar_W"]

        # Reshape.
        hit_masks = nvs_frame["hit_dict"]["masks"]
        hit_masks = hit_masks.reshape((lidar_H, lidar_W))
        hit_depths = nvs_frame["hit_dict"]["depths"]
        hit_depths[hit_depths == np.inf] = 0
        hit_depths = hit_depths.reshape((lidar_H, lidar_W))
        hit_normals = hit_normals.reshape((lidar_H, lidar_W, 3))
        hit_incidences = hit_incidences.reshape((lidar_H, lidar_W))
        intensities = nvs_frame["intensities"]
        intensities = intensities.reshape((lidar_H, lidar_W))
        rays_o = rays_o.reshape((lidar_H, lidar_W, 3))
        rays_d = rays_d.reshape((lidar_H, lidar_W, 3))
        raydrop_masks = raydrop_masks.reshape((lidar_H, lidar_W))

        # Cast.
        hit_masks = hit_masks.astype(np.float32)
        hit_depths = hit_depths.astype(np.float32)
        hit_normals = hit_normals.astype(np.float32)
        hit_incidences = hit_incidences.astype(np.float32)
        intensities = intensities.astype(np.float32)
        rays_o = rays_o.astype(np.float32)
        rays_d = rays_d.astype(np.float32)
        raydrop_masks = raydrop_masks.astype(np.float32)

        raydrop_datum = {
            "hit_masks": hit_masks,
            "hit_depths": hit_depths,
            "hit_normals": hit_normals,
            "hit_incidences": hit_incidences,
            "intensities": intensities,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "raydrop_masks": raydrop_masks,
        }
        raydrop_data.append(raydrop_datum)

    return raydrop_data
