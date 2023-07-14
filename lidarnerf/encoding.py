import torch
import torch.nn as nn
import numpy as np


class FreqEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        max_freq_log2,
        N_freqs,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):
        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out


def get_encoder(
    encoding,
    input_dim=3,
    multires=6,
    degree=4,
    num_levels=16,
    level_dim=2,
    base_resolution=16,
    log2_hashmap_size=19,
    desired_resolution=2048,
    align_corners=False,
    **kwargs
):
    if encoding == "None":
        return lambda x, **kwargs: x, input_dim

    elif encoding == "frequency":
        # encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        from freqencoder import FreqEncoder

        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == "sphere_harmonics":
        from shencoder import SHEncoder

        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == "hashgrid":
        from gridencoder import GridEncoder

        encoder = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
            gridtype="hash",
            align_corners=align_corners,
        )

    elif encoding == "tiledgrid":
        from gridencoder import GridEncoder

        encoder = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
            gridtype="tiled",
            align_corners=align_corners,
        )

    elif encoding == "ash":
        from ashencoder import AshEncoder

        encoder = AshEncoder(
            input_dim=input_dim,
            output_dim=16,
            log2_hashmap_size=log2_hashmap_size,
            resolution=desired_resolution,
        )

    else:
        raise NotImplementedError(
            "Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]"
        )

    return encoder, encoder.output_dim


class PeriodicVolumeEncoding(nn.Module):
    """Periodic Volume encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        smoothstep: bool = False,
    ) -> None:
        super(PeriodicVolumeEncoding, self).__init__()
        self.in_dim = 3
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        assert log2_hashmap_size % 3 == 0
        self.hash_table_size = 2**log2_hashmap_size
        self.n_output_dims = num_levels * features_per_level
        self.smoothstep = smoothstep

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.periodic_volume_resolution = 2 ** (log2_hashmap_size // 3)
        # self.periodic_resolution = torch.minimum(torch.floor(self.scalings), periodic_volume_resolution)

        self.hash_offset = levels * self.hash_table_size
        self.hash_table = (
            torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2
            - 1
        )
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

        # TODO weight loss by level?
        self.per_level_weights = 1.0

    def parameters(self):
        return self.hash_table

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor):
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # round to make it perioidic
        x = in_tensor
        x %= self.periodic_volume_resolution
        # xyz to index
        x = (
            x[..., 0] * (self.periodic_volume_resolution**2)
            + x[..., 1] * (self.periodic_volume_resolution)
            + x[..., 2]
        )
        # offset by feature levels
        x += self.hash_offset.to(x.device)

        return x.long()

    def pytorch_fwd(self, in_tensor):
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(
            in_tensor.device
        )  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        if self.smoothstep:
            offset = offset * offset * (3.0 - 2.0 * offset)

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(
            torch.cat(
                [scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1
            )
        )
        hashed_2 = self.hash_fn(
            torch.cat(
                [scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1
            )
        )
        hashed_3 = self.hash_fn(
            torch.cat(
                [scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1
            )
        )
        hashed_4 = self.hash_fn(
            torch.cat(
                [scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1
            )
        )
        hashed_5 = self.hash_fn(
            torch.cat(
                [scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1
            )
        )
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(
            torch.cat(
                [scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1
            )
        )

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(
            encoded_value, start_dim=-2, end_dim=-1
        )  # [..., num_levels * features_per_level]

    def forward(self, in_tensor):
        return self.pytorch_fwd(in_tensor)

    def get_total_variation_loss(self):
        """Compute the total variation loss for the feature volume."""
        feature_volume = self.hash_table.reshape(
            self.num_levels,
            self.periodic_volume_resolution,
            self.periodic_volume_resolution,
            self.periodic_volume_resolution,
            self.features_per_level,
        )
        diffx = feature_volume[:, 1:, :, :, :] - feature_volume[:, :-1, :, :, :]
        diffy = feature_volume[:, :, 1:, :, :] - feature_volume[:, :, :-1, :, :]
        diffz = feature_volume[:, :, :, 1:, :] - feature_volume[:, :, :, :-1, :]

        # TODO how to sum here or should we use mask?
        resx = diffx.abs().mean(dim=(1, 2, 3, 4))
        resy = diffy.abs().mean(dim=(1, 2, 3, 4))
        resz = diffz.abs().mean(dim=(1, 2, 3, 4))

        return ((resx + resy + resz) * self.per_level_weights).mean()
