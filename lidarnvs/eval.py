import numpy as np
import torch

from skimage.metrics import structural_similarity
from extern.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from extern.fscore import fscore


def eval_points_and_pano(
    gt_local_points: np.ndarray,
    pd_local_points: np.ndarray,
    gt_intensities: np.ndarray,
    pd_intensities: np.ndarray,
    gt_pano: np.ndarray,
    pd_pano: np.ndarray,
) -> dict:
    """
    Args:
        gt_local_points: (N, 3), float32, local point coords in world-scale.
        pd_local_points: (M, 3), float32, local point coords in world-scale.
        gt_intensities: (H, W), float32, point intensities, >= 0.
        pd_intensities: (H, W), float32, point intensities, >= 0.
        gt_pano: (H, W), float32, range depth image in world-scale.
            0 means dropped rays. A dropped ray must not have intensity.
        pd_pano: (H, W), float32, range depth image in world-scale.
            0 means dropped rays. A dropped ray must not have intensity.

    Returns:
        # Depth metrics
        - metrics["depth_rmse"]
        - metrics["depth_a1"]
        - metrics["depth_a2"]
        - metrics["depth_a3"]
        # Point metrics
        - metrics["chamfer"]
        - metrics["f_score"]
        # Intensity metrics
        - metrics["intensity_mae"]
    """
    # Sanity checks.
    if not gt_local_points.ndim == 2 or not gt_local_points.shape[1] == 3:
        raise ValueError(
            f"gt_local_points must be (N, 3), but got {gt_local_points.shape}"
        )
    if not pd_local_points.ndim == 2 or not pd_local_points.shape[1] == 3:
        raise ValueError(
            f"pd_local_points must be (M, 3), but got {pd_local_points.shape}"
        )
    if not gt_intensities.ndim == 2:
        raise ValueError(
            f"gt_intensities must be (H, W), but got {gt_intensities.shape}"
        )
    lidar_H, lidar_W = gt_intensities.shape
    if not pd_intensities.shape == (lidar_H, lidar_W):
        raise ValueError(
            f"pd_intensities must be (H, W), but got {pd_intensities.shape}"
        )
    if not gt_pano.shape == (lidar_H, lidar_W):
        raise ValueError(f"gt_pano must be (H, W), but got {gt_pano.shape}")
    if not pd_pano.shape == (lidar_H, lidar_W):
        raise ValueError(f"pd_pano must be (H, W), but got {pd_pano.shape}")

    # All shall be numpy array
    is_instance_all = [
        isinstance(e, np.ndarray)
        for e in [
            gt_local_points,
            pd_local_points,
            gt_intensities,
            pd_intensities,
            gt_pano,
            pd_pano,
        ]
    ]
    if not all(is_instance_all):
        raise ValueError("All inputs must be numpy array.")

    def compute_depth_metrics(
        gt_depths, pd_depths, min_depth=1e-3, max_depth=80, thresh_set=1.25
    ):
        pd_depths[pd_depths < min_depth] = min_depth
        pd_depths[pd_depths > max_depth] = max_depth
        gt_depths[gt_depths < min_depth] = min_depth
        gt_depths[gt_depths > max_depth] = max_depth

        thresh = np.maximum((gt_depths / pd_depths), (pd_depths / gt_depths))
        a1 = (thresh < thresh_set).mean()
        a2 = (thresh < thresh_set**2).mean()
        a3 = (thresh < thresh_set**3).mean()
        rmse = (gt_depths - pd_depths) ** 2
        rmse = np.sqrt(rmse.mean())
        ssim = structural_similarity(
            gt_depths,
            pd_depths,
            data_range=gt_depths.max() - gt_depths.min(),
        )
        return rmse, a1, a2, a3, ssim

    def compute_point_metrics(gt_points, pd_points):
        chamLoss = chamfer_3DDist()
        dist1, dist2, idx1, idx2 = chamLoss(
            torch.tensor(pd_points[None, ...]).float().cuda(),
            torch.tensor(gt_points[None, ...]).float().cuda(),
        )
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)

        chamfer_dis = chamfer_dis.item()
        f_score = f_score.item()
        return chamfer_dis, f_score

    def compute_intensity_metrics(gt_intensities, pd_intensities):
        mae = np.abs(gt_intensities - pd_intensities).mean()
        return mae

    metrics = dict()
    (
        metrics["depth_rmse"],
        metrics["depth_a1"],
        metrics["depth_a2"],
        metrics["depth_a3"],
        metrics["depth_ssim"],
    ) = compute_depth_metrics(gt_depths=gt_pano.flatten(), pd_depths=pd_pano.flatten())

    (
        metrics["chamfer"],
        metrics["f_score"],
    ) = compute_point_metrics(gt_points=gt_local_points, pd_points=pd_local_points)

    metrics["intensity_mae"] = compute_intensity_metrics(
        gt_intensities=gt_intensities, pd_intensities=pd_intensities
    )

    return metrics
