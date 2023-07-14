import camtools as ct
import numpy as np
import torch
from tqdm import tqdm

from lidarnerf.convert import (
    lidar_to_pano_with_intensities,
    lidar_to_pano_with_intensities_fpa,
    pano_to_lidar_with_intensities,
)
from lidarnvs.loader import extract_dataset_frame
from lidarnvs.raydrop_train_pcgen import RayDrop, run_network, get_embedder
from lidarnvs.lidarnvs_base import LidarNVSBase


class LidarNVSPCGen(LidarNVSBase):
    def __init__(self, raycasting="cp", ckpt_path=None):
        self.raycasting = raycasting

        # Network for predicting raydrop.
        if ckpt_path is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embed_fn, input_ch = get_embedder(4, input_dims=1, i=-1)
            self.embeddirs_fn, input_ch_views = get_embedder(10, input_dims=3, i=-1)
            total_input_ch = input_ch * 2 + input_ch_views
            netdepth, netwidth = 4, 128
            self.model = RayDrop(D=netdepth, W=netwidth, input_ch=total_input_ch).to(
                self.device
            )

            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt["network_fn_state_dict"])
            self.model.eval()
            print(f"Checkpoint loaded from {ckpt_path}")

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

        # Save points and intensities for interpolation.
        self.points = all_points
        self.point_intensities = all_point_intensities

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
        local_points = ct.project.homo_project(
            self.points,
            ct.convert.pose_to_T(lidar_pose),
        )

        # Pano intensities.
        local_points_with_intensities = np.concatenate(
            [local_points, self.point_intensities.reshape((-1, 1))], axis=1
        )
        if self.raycasting == "cp":
            pano, intensities = lidar_to_pano_with_intensities(
                local_points_with_intensities=local_points_with_intensities,
                lidar_H=lidar_H,
                lidar_W=lidar_W,
                lidar_K=lidar_K,
            )
        elif self.raycasting == "fpa":
            pano, intensities = lidar_to_pano_with_intensities_fpa(
                local_points_with_intensities=local_points_with_intensities,
                lidar_H=lidar_H,
                lidar_W=lidar_W,
                lidar_K=lidar_K,
            )

        local_points_with_intensities = pano_to_lidar_with_intensities(
            pano=pano, intensities=intensities, lidar_K=lidar_K
        )
        local_points = local_points_with_intensities[:, :3]
        local_point_intensities = local_points_with_intensities[:, 3]

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
        nvs_frame = self.predict_frame(
            lidar_K=lidar_K,
            lidar_pose=lidar_pose,
            lidar_H=lidar_H,
            lidar_W=lidar_W,
        )
        direction = get_direction(lidar_H, lidar_W, lidar_K)
        pano = nvs_frame["pano"]
        intensity = nvs_frame["intensities"]
        rays_val = np.concatenate(
            (
                np.array(direction).reshape(-1, 3),
                np.array(pano).reshape(-1, 1),
                np.array(intensity).reshape(-1, 1),
            ),
            -1,
        )
        rays_val = torch.Tensor(rays_val).to(self.device)
        pd_raydrop_masks = run_network(
            rays_val, self.model, self.embed_fn, self.embeddirs_fn
        )
        pd_raydrop_masks = np.where(
            pd_raydrop_masks.cpu().numpy() > 0.5, 1.0, 0.0
        ).reshape(lidar_H, lidar_W)
        # Update predict_dict
        # Frame properties.
        pano = nvs_frame["pano"]
        intensities = nvs_frame["intensities"]
        if not np.all(pd_raydrop_masks == 0):
            pano = pano * pd_raydrop_masks
            intensities = intensities * pd_raydrop_masks
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
        }

        return predict_dict


def generate_raydrop_data_pcgen(dataset, nvs: LidarNVSPCGen, rm_pano_mask=True) -> dict:
    """
    Prepare dataset for learning ray drop.
    The frames are NOT loaded by our dataset, but GENERATED.

    Return:
        directions, panos, intensities, raydrop_masks
    """

    raydrop_masks = []
    directions = []
    panos = []
    intensities = []
    for frame_idx in tqdm(range(len(dataset)), desc="Prepare raydrop dataset"):
        gt_frame = extract_dataset_frame(
            dataset, frame_idx=frame_idx, rm_pano_mask=rm_pano_mask
        )
        nvs_frame = nvs.predict_frame(
            lidar_K=gt_frame["lidar_K"],
            lidar_pose=gt_frame["lidar_pose"],
            lidar_H=gt_frame["lidar_H"],
            lidar_W=gt_frame["lidar_W"],
        )

        # The target.
        raydrop_masks.append(gt_frame["pano"])

        # The inputs
        lidar_H, lidar_W, lidar_K = (
            gt_frame["lidar_H"],
            gt_frame["lidar_W"],
            gt_frame["lidar_K"],
        )
        directions.append(get_direction(lidar_H, lidar_W, lidar_K))
        panos.append(nvs_frame["pano"])
        intensities.append(nvs_frame["intensities"])
    return (directions, panos, intensities, raydrop_masks)


def get_direction(lidar_H, lidar_W, lidar_K):
    fov_up, fov = lidar_K
    i, j = np.meshgrid(
        np.arange(lidar_W, dtype=np.float32),
        np.arange(lidar_H, dtype=np.float32),
        indexing="xy",
    )
    beta = -(i - lidar_W / 2) / lidar_W * 2 * np.pi
    alpha = (fov_up - j / lidar_H * fov) / 180 * np.pi
    dirs = np.stack(
        [np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta), np.sin(alpha)], -1
    )
    return dirs
