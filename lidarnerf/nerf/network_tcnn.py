import torch

import numpy as np

import tinycudann as tcnn
from lidarnerf.activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        encoding="HashGrid",
        desired_resolution=2048,
        log2_hashmap_size=19,
        encoding_dir="SphericalHarmonics",
        n_features_per_level=2,
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        out_color_dim=3,
        out_lidar_color_dim=2,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.desired_resolution = desired_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.out_color_dim = out_color_dim
        self.out_lidar_color_dim = out_lidar_color_dim
        self.n_features_per_level = n_features_per_level

        per_level_scale = np.exp2(
            np.log2(self.desired_resolution * bound / 16) / (16 - 1)
        )
        print(f"TCNN desired resolution: {self.desired_resolution}")
        print(f"TCNN per level scale: {per_level_scale}")

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
                # "interpolation": "Smoothstep"
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        # # SH
        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        # # Hash
        # per_level_scale = np.exp2(np.log2(1024 * bound / 4) / (4 - 1))
        # self.encoder_dir = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 4,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": self.log2_hashmap_size,
        #         "base_resolution": 128,
        #         "per_level_scale": per_level_scale,
        #     },
        # )
        # # freq
        self.encoder_lidar_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 12,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=self.out_color_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        self.in_dim_lidar_color = (
            self.encoder_lidar_dir.n_output_dims + self.geo_feat_dim
        )
        self.lidar_color_net = tcnn.Network(
            n_input_dims=self.in_dim_lidar_color,
            n_output_dims=self.out_lidar_color_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def forward(self, x, d):
        pass

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
        }

    # allow masked inference
    def color(self, x, d, cal_lidar_color=False, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(
                mask.shape[0], self.out_dim, dtype=x.dtype, device=x.device
            )  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        # d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        # d = self.encoder_dir(d)

        # h = torch.cat([d, geo_feat], dim=-1)
        if cal_lidar_color:
            d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
            d = self.encoder_lidar_dir(d)
            h = torch.cat([d, geo_feat], dim=-1)
            h = self.lidar_color_net(h)
        else:
            d = (d + 1) / 2
            d = self.encoder_dir(d)
            h = torch.cat([d, geo_feat], dim=-1)
            h = self.color_net(h)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    # optimizer utils
    def get_params(self, lr):
        params = [
            {"params": self.encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.encoder_dir.parameters(), "lr": lr},
            {"params": self.encoder_lidar_dir.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
            {"params": self.lidar_color_net.parameters(), "lr": lr},
        ]
        if self.bg_radius > 0:
            params.append({"params": self.encoder_bg.parameters(), "lr": lr})
            params.append({"params": self.bg_net.parameters(), "lr": lr})

        return params
