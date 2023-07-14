import camtools as ct
import numpy as np

from lidarnerf.dataset.base_dataset import get_lidar_rays
from lidarnerf.convert import pano_to_lidar_with_intensities


def extract_dataset_frame(
    dataset, frame_idx: int, rm_pano_mask: bool = True, verbose: bool = False
) -> dict:
    """
    Extract a single frame from a dataset object.
    """
    # Unpack dataset.
    lidar_pose = dataset.poses_lidar[frame_idx].numpy()
    pano = dataset.images_lidar[frame_idx][:, :, 2].numpy()
    intensities = dataset.images_lidar[frame_idx][:, :, 1].numpy()
    lidar_K = dataset.intrinsics_lidar
    lidar_H = dataset.H_lidar
    lidar_W = dataset.W_lidar

    # Process pano mask.
    # TODO: remove this.
    pano_mask = pano != -1
    if rm_pano_mask:
        pano[pano == -1] = 0

    # Load rays.
    ray_dict = get_lidar_rays(
        poses=dataset.poses_lidar[[frame_idx]],
        intrinsics=dataset.intrinsics_lidar,
        H=dataset.H_lidar,
        W=dataset.W_lidar,
        N=-1,
        patch_size=1,
    )
    rays_o = ray_dict["rays_o"].squeeze().numpy()
    rays_d = ray_dict["rays_d"].squeeze().numpy()
    rays = np.concatenate([rays_o, rays_d], axis=-1)

    # Generate gt data.
    # pose: cam to world projection matrix.
    # T   : world to cam projection matrix.
    # (N, 4)
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=intensities,
        lidar_K=lidar_K,
    )
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]

    # Project local to world coordinates.
    points = ct.project.homo_project(local_points, lidar_pose)
    point_intensities = local_point_intensities

    # "pano"       : invalid points marked as 0 (depth).
    # "intensities": 0 means likely invalid, but not 100%.
    frame_dict = {
        "rays": rays,
        "lidar_pose": lidar_pose,
        "lidar_K": lidar_K,
        "lidar_H": lidar_H,
        "lidar_W": lidar_W,
        # Frame properties.
        "pano": pano,
        "pano_mask": pano_mask,
        "intensities": intensities,
        # Local coord properties.
        "local_points": local_points,
        "local_point_intensities": local_point_intensities,
        # World coord properties.
        "points": points,
        "point_intensities": point_intensities,
    }
    if verbose:
        for key, val in frame_dict.items():
            if isinstance(val, np.ndarray):
                print(f"- {key}: {val.shape}")
            else:
                print(f"- {key}: {val}")

    return frame_dict
