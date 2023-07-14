import math
import trimesh

import torch
import torch.nn as nn

from lidarnerf import raymarching


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples
        ).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print("[visualize points]", pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(
        self,
        bound=1,
        density_scale=1,  # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
        min_near=0.2,
        min_near_lidar=0.2,
        density_thresh=0.01,
        bg_radius=-1,
    ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.min_near_lidar = min_near_lidar
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius  # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        rays_o,
        rays_d,
        cal_lidar_color=False,
        num_steps=128,
        upsample_steps=128,
        bg_color=None,
        perturb=False,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
        if cal_lidar_color:
            self.out_dim = self.out_lidar_color_dim
        else:
            self.out_dim = self.out_color_dim

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        if cal_lidar_color:
            nears = (
                torch.ones(N, dtype=rays_o.dtype, device=rays_o.device)
                * self.min_near_lidar
            )
            fars = (
                torch.ones(N, dtype=rays_o.dtype, device=rays_o.device)
                * self.min_near_lidar
                * 81.0
            )  # hard code
        else:
            nears, fars = raymarching.near_far_from_aabb(
                rays_o, rays_d, aabb, self.min_near
            )

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(
            0
        )  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = (
                z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            )
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1
        )  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat(
                    [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1
                )

                alphas = 1 - torch.exp(
                    -deltas * self.density_scale * density_outputs["sigma"].squeeze(-1)
                )  # [N, T]
                alphas_shifted = torch.cat(
                    [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1
                )  # [N, T+1]
                weights = (
                    alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
                )  # [N, T]

                z_vals_mid = z_vals[..., :-1] + 0.5 * deltas[..., :-1]  # [N, T-1]
                new_z_vals = sample_pdf(
                    z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training
                ).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(
                    -2
                ) * new_z_vals.unsqueeze(
                    -1
                )  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(
                    torch.max(new_xyzs, aabb[:3]), aabb[3:]
                )  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            # new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(
                xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs)
            )

            for k in density_outputs:
                tmp_output = torch.cat(
                    [density_outputs[k], new_density_outputs[k]], dim=1
                )
                density_outputs[k] = torch.gather(
                    tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output)
                )

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat(
            [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1
        )
        alphas = 1 - torch.exp(
            -deltas * self.density_scale * density_outputs["sigma"].squeeze(-1)
        )  # [N, T+t]
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1
        )  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        rgbs = self.color(
            xyzs.reshape(-1, 3),
            dirs.reshape(-1, 3),
            cal_lidar_color=cal_lidar_color,
            mask=mask.reshape(-1),
            **density_outputs
        )

        rgbs = rgbs.view(N, -1, self.out_dim)  # [N, T+t, 3]

        # print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth  Note: not real depth!!
        # ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        # depth = torch.sum(weights * ori_z_vals, dim=-1)
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(
                rays_o, rays_d, self.bg_radius
            )  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        if not cal_lidar_color:
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, self.out_dim)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

        return {
            "depth_lidar": depth,
            "image_lidar": image,
            "weights_sum_lidar": weights_sum,
        }

    def render(
        self,
        rays_o,
        rays_d,
        cal_lidar_color=False,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if staged:
            if cal_lidar_color:
                out_dim = self.out_lidar_color_dim
                res_keys = ["depth_lidar", "image_lidar"]
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, out_dim), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        cal_lidar_color=cal_lidar_color,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_[res_keys[0]]
                    image[b : b + 1, head:tail] = results_[res_keys[1]]
                    head += max_ray_batch

            results = {}
            results[res_keys[0]] = depth
            results[res_keys[1]] = image

        else:
            results = _run(rays_o, rays_d, cal_lidar_color=cal_lidar_color, **kwargs)

        return results
