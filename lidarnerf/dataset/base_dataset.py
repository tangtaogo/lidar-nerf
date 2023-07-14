import numpy as np
import torch
import trimesh
from packaging import version as pver
from dataclasses import dataclass


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


@torch.cuda.amp.autocast(enabled=False)
def get_lidar_rays(poses, intrinsics, H, W, N=-1, patch_size=1):
    """
    Get lidar rays.

    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [2]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """
    device = poses.device
    B = poses.shape[0]

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float
    # i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    # j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    i = i.t().reshape([1, H * W]).expand([B, H * W])
    j = j.t().reshape([1, H * W]).expand([B, H * W])
    results = {}
    if N > 0:
        N = min(N, H * W)

        if isinstance(patch_size, int):
            patch_size_x, patch_size_y = patch_size, patch_size
        elif len(patch_size) == 1:
            patch_size_x, patch_size_y = patch_size[0], patch_size[0]
        else:
            patch_size_x, patch_size_y = patch_size

        if patch_size_x > 0:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner
            # pixels... but I don't have other ideas.
            num_patch = N // (patch_size_x * patch_size_y)
            inds_x = torch.randint(0, H - patch_size_x, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size_y, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(
                torch.arange(patch_size_x, device=device),
                torch.arange(patch_size_y, device=device),
            )
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        else:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
        results["inds"] = inds

    fov_up, fov = intrinsics
    beta = -(i - W / 2) / W * 2 * np.pi
    alpha = (fov_up - j / H * fov) / 180 * np.pi

    directions = torch.stack(
        [
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta),
            torch.sin(alpha),
        ],
        -1,
    )
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1):
    """get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}
    if N > 0:
        N = min(N, H * W)

        if patch_size > 1:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner
            # pixels... but I don't have other ideas.
            num_patch = N // (patch_size**2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(
                torch.arange(patch_size, device=device),
                torch.arange(patch_size, device=device),
            )
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        else:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array(
            [
                [pos, a],
                [pos, b],
                [pos, c],
                [pos, d],
                [a, b],
                [b, c],
                [c, d],
                [d, a],
                [pos, o],
            ]
        )
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


@dataclass
class BaseDataset:
    pass
