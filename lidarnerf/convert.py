import numpy as np


def lidar_to_pano_with_intensities_with_bbox_mask(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    bbox_local: np.ndarray,
    max_depth=80,
    max_intensity=255.0,
):
    """
    Convert lidar frame to pano frame with intensities with bbox_mask.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        bbox_local: (8x4), world bbox in local.
        max_depth: max depth in meters.
        max_intensity: max intensity.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """

    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))

    # bbox mask
    pano[:, :] = -1
    r_min, r_max, c_min, c_max = 1e5, -1, 1e5, -1
    for bbox_local_point in bbox_local:
        x, y, z, _ = bbox_local_point
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi

        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue
        else:
            r_min, r_max, c_min, c_max = (
                min(r_min, r),
                max(r_max, r),
                min(c_min, c),
                max(c_max, c),
            )

    pano[r_min:r_max, c_min:c_max] = 0

    # Fill pano and intensities.
    for local_points, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity / max_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity / max_intensity

    return pano, intensities


def lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    max_depth=80,
):
    """
    Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """
    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for local_points, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity

    return pano, intensities


def lidar_to_pano(
    local_points: np.ndarray, lidar_H: int, lidar_W: int, lidar_K: int, max_dpeth=80
):
    """
    Convert lidar frame to pano frame. Lidar points are in local coordinates.

    Args:
        local_points: (N, 3), float32, in lidar frame.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
    """

    # (N, 3) -> (N, 4), filled with zeros.
    local_points_with_intensities = np.concatenate(
        [local_points, np.zeros((local_points.shape[0], 1))], axis=1
    )
    pano, _ = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=lidar_K,
        max_dpeth=max_dpeth,
    )
    return pano


def pano_to_lidar_with_intensities(pano: np.ndarray, intensities, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        intensities: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points_with_intensities: (N, 4), float32, in lidar frame.
    """
    fov_up, fov = lidar_K

    H, W = pano.shape
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    beta = -(i - W / 2) / W * 2 * np.pi
    alpha = (fov_up - j / H * fov) / 180 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
    local_points = dirs * pano.reshape(H, W, 1)

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H, W, 1)], axis=2
    )

    # Filter empty points.
    idx = np.where(pano != 0.0)
    local_points_with_intensities = local_points_with_intensities[idx]

    return local_points_with_intensities


def pano_to_lidar(pano, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
    )
    return local_points_with_intensities[:, :3]


def lidar_to_pano_with_intensities_fpa(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    max_depth=80,
    z_buffer_len=10,
):
    """
    Convert lidar frame to pano frame with intensities with bbox_mask.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.
        z_buffer_len: length of the z_buffer.

    Return:
        rangeview image: (H, W, 3), float32.
    """

    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    range_view = np.zeros((lidar_H, lidar_W, 3, z_buffer_len + 1))

    for local_point, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_point
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        position = range_view[r, c, 2, 0] + 1
        if position > z_buffer_len:
            depth_z_buffer = list(range_view[r, c, 2][1:]) + [dist]
            intensity_z_buffer = list(range_view[r, c, 1][1:]) + [local_point_intensity]
            position = position - 1

            sort_index = np.argsort(depth_z_buffer)
            depth_z_buffer = np.insert(
                np.array(depth_z_buffer)[sort_index][:z_buffer_len], 0, position
            )
            intensity_z_buffer = np.insert(
                np.array(intensity_z_buffer)[sort_index][:z_buffer_len], 0, position
            )
            range_view[r, c, 2] = depth_z_buffer
            range_view[r, c, 1] = intensity_z_buffer

        else:
            range_view[r, c, 2, int(position)] = dist
            range_view[r, c, 1, int(position)] = local_point_intensity
        range_view[r, c, 2, 0] = position
    range_view = parse_z_buffer(range_view, lidar_H, lidar_W)
    return range_view[:, :, 2], range_view[:, :, 1]


def parse_z_buffer(novel_pano, lidar_H, lidar_W, threshold=0.2):
    range_view = np.zeros((lidar_H, lidar_W, 3))
    for i in range(lidar_H):
        for j in range(lidar_W):
            range_pixel = novel_pano[i, j, 2]
            intensity_pixel = novel_pano[i, j, 1]
            z_buffer_num = int(range_pixel[0])
            if z_buffer_num == 0:
                continue
            if z_buffer_num == 1:
                range_view[i][j][2] = range_pixel[1]
                range_view[i][j][1] = intensity_pixel[1]
                continue

            depth_z_buffer = range_pixel[1:z_buffer_num]
            cloest_points = min(depth_z_buffer)
            index = depth_z_buffer <= (cloest_points + threshold)

            final_depth_z_buffer = np.array(depth_z_buffer)[index]
            final_dis = np.average(
                final_depth_z_buffer, weights=1 / final_depth_z_buffer
            )
            range_view[i][j][2] = final_dis

            intensity_z_buffer = intensity_pixel[1:z_buffer_num]
            final_intensity_z_buffer = np.array(intensity_z_buffer)[index]
            final_intensity = np.average(
                final_intensity_z_buffer, weights=1 / final_depth_z_buffer
            )
            range_view[i][j][1] = final_intensity
    return range_view
