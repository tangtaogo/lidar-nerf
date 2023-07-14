import torch
import configargparse
import os
import numpy as np

from lidarnerf.nerf.utils import (
    seed_everything,
    RMSEMeter,
    MAEMeter,
    DepthMeter,
    PointsMeter,
    Trainer,
)


def get_arg_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument(
        "--config",
        is_config_file=True,
        default="configs/kitti360_1908.txt",
        help="config file path",
    )
    parser.add_argument("--path", type=str, default="data/kitti360")
    parser.add_argument(
        "-L", action="store_true", help="equals --fp16 --tcnn --preload"
    )
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--test_eval", action="store_true", help="test and eval mode")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument(
        "--cluster_summary_path",
        type=str,
        default="/summary",
        help="Overwrite default summary path if on cluster",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataloader", type=str, choices=("kitti360", "nerf_mvl"), default="kitti360"
    )
    parser.add_argument("--sequence_id", type=str, default="1908")

    ### lidar-nerf
    parser.add_argument("--enable_lidar", action="store_true", help="Enable lidar.")
    parser.add_argument("--alpha_d", type=float, default=1e3)
    parser.add_argument("--alpha_r", type=float, default=1)
    parser.add_argument("--alpha_i", type=float, default=1)
    parser.add_argument("--alpha_grad_norm", type=float, default=1)
    parser.add_argument("--alpha_spatial", type=float, default=0.1)
    parser.add_argument("--alpha_tv", type=float, default=1)
    parser.add_argument("--alpha_grad", type=float, default=1e2)

    parser.add_argument("--intensity_inv_scale", type=float, default=1)

    parser.add_argument("--spatial_smooth", action="store_true")
    parser.add_argument("--grad_norm_smooth", action="store_true")
    parser.add_argument("--tv_loss", action="store_true")
    parser.add_argument("--grad_loss", action="store_true")
    parser.add_argument("--sobel_grad", action="store_true")

    parser.add_argument(
        "--desired_resolution",
        type=int,
        default=2048,
        help="TCN finest resolution at the smallest scale",
    )
    parser.add_argument("--log2_hashmap_size", type=int, default=19)
    parser.add_argument("--n_features_per_level", type=int, default=2)
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num_layers of sigmanet"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="hidden_dim of sigmanet"
    )
    parser.add_argument(
        "--geo_feat_dim", type=int, default=15, help="geo_feat_dim of sigmanet"
    )
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument(
        "--num_rays_lidar",
        type=int,
        default=4096,
        help="num rays sampled per image for each training step",
    )
    parser.add_argument(
        "--min_near_lidar",
        type=float,
        default=0.01,
        help="minimum near distance for camera",
    )
    parser.add_argument(
        "--depth_loss", type=str, default="l1", help="l1, bce, mse, huber"
    )
    parser.add_argument(
        "--depth_grad_loss", type=str, default="l1", help="l1, bce, mse, huber"
    )
    parser.add_argument(
        "--intensity_loss", type=str, default="mse", help="l1, bce, mse, huber"
    )
    parser.add_argument(
        "--raydrop_loss", type=str, default="mse", help="l1, bce, mse, huber"
    )
    parser.add_argument(
        "--patch_size_lidar",
        type=int,
        default=1,
        help="[experimental] render patches in training. "
        "1 means disabled, use [64, 32, 16] to enable",
    )
    parser.add_argument(
        "--change_patch_size_lidar",
        nargs="+",
        type=int,
        default=[1, 1],
        help="[experimental] render patches in training. "
        "1 means disabled, use [64, 32, 16] to enable, change during training",
    )
    parser.add_argument(
        "--change_patch_size_epoch",
        type=int,
        default=2,
        help="change patch_size intenvel",
    )

    ### training options
    parser.add_argument(
        "--iters",
        type=int,
        default=30000,
        help="training iters",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument(
        "--num_rays",
        type=int,
        default=4096,
        help="num rays sampled per image for each training step",
    )
    parser.add_argument(
        "--num_steps", type=int, default=768, help="num steps sampled per ray"
    )
    parser.add_argument(
        "--upsample_steps", type=int, default=64, help="num steps up-sampled per ray"
    )
    parser.add_argument(
        "--max_ray_batch",
        type=int,
        default=4096,
        help="batch size of rays at inference to avoid OOM)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1,
        help="[experimental] render patches in training, so as to apply "
        "LPIPS loss. 1 means disabled, use [64, 32, 16] to enable",
    )

    ### network backbone options
    parser.add_argument(
        "--fp16", action="store_true", help="use amp mixed precision training"
    )
    parser.add_argument("--tcnn", action="store_true", help="use TCNN backend")

    ### dataset options
    parser.add_argument(
        "--color_space",
        type=str,
        default="srgb",
        help="Color space, supports (linear, srgb)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="preload all data into GPU, accelerate training but use more GPU memory",
    )
    # (the default value is for the fox dataset)
    parser.add_argument(
        "--bound",
        type=float,
        default=2,
        help="assume the scene is bounded in box[-bound, bound]^3, "
        "if > 1, will invoke adaptive ray marching.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.33,
        help="scale camera location into box[-bound, bound]^3",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs="*",
        default=[0, 0, 0],
        help="offset of camera location",
    )
    parser.add_argument(
        "--dt_gamma",
        type=float,
        default=1 / 128,
        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 "
        "to accelerate rendering (but usually with worse quality)",
    )
    parser.add_argument(
        "--min_near", type=float, default=0.2, help="minimum near distance for camera"
    )
    parser.add_argument(
        "--density_thresh",
        type=float,
        default=10,
        help="threshold for density grid to be occupied",
    )
    parser.add_argument(
        "--bg_radius",
        type=float,
        default=-1,
        help="if positive, use a background model at sphere(bg_radius)",
    )

    return parser


def main():
    parser = get_arg_parser()
    opt = parser.parse_args()
    opt.enable_lidar = True

    # Check sequence id.
    kitti360_sequence_ids = [
        "1538",
        "1728",
        "1908",
        "3353",
    ]
    nerf_mvl_sequence_ids = [
        "bollard",
        "car",
        "pedestrian",
        "pier",
        "plant",
        "tire",
        "traffic_cone",
        "warning_sign",
        "water_safety_barrier",
    ]

    # Specify dataloader class
    if opt.dataloader == "kitti360":
        from lidarnerf.dataset.kitti360_dataset import KITTI360Dataset as NeRFDataset

        if opt.sequence_id not in kitti360_sequence_ids:
            raise ValueError(
                f"Unknown sequence id {opt.sequence_id} for {opt.dataloader}"
            )
    elif opt.dataloader == "nerf_mvl":
        from lidarnerf.dataset.nerfmvl_dataset import NeRFMVLDataset as NeRFDataset

        if opt.sequence_id not in nerf_mvl_sequence_ids:
            raise ValueError(
                f"Unknown sequence id {opt.sequence_id} for {opt.dataloader}"
            )
    else:
        raise RuntimeError("Should not reach here.")

    os.makedirs(opt.workspace, exist_ok=True)
    f = os.path.join(opt.workspace, "args.txt")
    with open(f, "w") as file:
        for arg in vars(opt):
            attr = getattr(opt, arg)
            file.write("{} = {}\n".format(arg, attr))

    if opt.L:
        opt.fp16 = True
        opt.tcnn = True
        opt.preload = True

    if opt.patch_size > 1:
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert (
            opt.num_rays % (opt.patch_size**2) == 0
        ), "patch_size ** 2 should be dividable by num_rays."

    opt.min_near = opt.scale  # hard-code, set min_near ori 1m
    opt.min_near_lidar = opt.scale

    if opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from lidarnerf.nerf.network_tcnn import NeRFNetwork

        model = NeRFNetwork(
            encoding="hashgrid",
            desired_resolution=opt.desired_resolution,
            log2_hashmap_size=opt.log2_hashmap_size,
            n_features_per_level=opt.n_features_per_level,
            num_layers=opt.num_layers,
            hidden_dim=opt.hidden_dim,
            geo_feat_dim=opt.geo_feat_dim,
            bound=opt.bound,
            density_scale=1,
            min_near=opt.min_near,
            min_near_lidar=opt.min_near_lidar,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        )
    else:
        from lidarnerf.nerf.network import NeRFNetwork

        model = NeRFNetwork(
            encoding="hashgrid",
            desired_resolution=opt.desired_resolution,
            log2_hashmap_size=opt.log2_hashmap_size,
            num_layers=opt.num_layers,
            hidden_dim=opt.hidden_dim,
            geo_feat_dim=opt.geo_feat_dim,
            bound=opt.bound,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        )

    print(opt)
    seed_everything(opt.seed)
    print(model)

    loss_dict = {
        "mse": torch.nn.MSELoss(reduction="none"),
        "l1": torch.nn.L1Loss(reduction="none"),
        "bce": torch.nn.BCEWithLogitsLoss(reduction="none"),
        "huber": torch.nn.HuberLoss(reduction="none", delta=0.2 * opt.scale),
        "cos": torch.nn.CosineSimilarity(),
    }
    criterion = {
        "depth": loss_dict[opt.depth_loss],
        "raydrop": loss_dict[opt.raydrop_loss],
        "intensity": loss_dict[opt.intensity_loss],
        "grad": loss_dict[opt.depth_grad_loss],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.test or opt.test_eval:
        test_loader = NeRFDataset(
            device=device,
            split="test",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            enable_lidar=opt.enable_lidar,
            num_rays_lidar=opt.num_rays_lidar,
        ).dataloader()
        if opt.enable_lidar:
            depth_metrics = [
                MAEMeter(intensity_inv_scale=opt.intensity_inv_scale),
                RMSEMeter(),
                DepthMeter(scale=opt.scale),
                PointsMeter(
                    scale=opt.scale, intrinsics=test_loader._data.intrinsics_lidar
                ),
            ]
        else:
            depth_metrics = []
        trainer = Trainer(
            "lidar_nerf",
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            depth_metrics=depth_metrics,
            use_checkpoint=opt.ckpt,
        )

        if test_loader.has_gt and opt.test_eval:
            trainer.evaluate(test_loader)  # blender has gt, so evaluate it.
        trainer.test(test_loader, write_video=False)  # test and save video
        trainer.save_mesh(resolution=128, threshold=10)

    else:
        optimizer = lambda model: torch.optim.Adam(
            model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15
        )

        train_loader = NeRFDataset(
            device=device,
            split="train",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            enable_lidar=opt.enable_lidar,
            num_rays_lidar=opt.num_rays_lidar,
        ).dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1)
        )
        if opt.enable_lidar:
            depth_metrics = [
                MAEMeter(intensity_inv_scale=opt.intensity_inv_scale),
                RMSEMeter(),
                DepthMeter(scale=opt.scale),
                PointsMeter(
                    scale=opt.scale, intrinsics=train_loader._data.intrinsics_lidar
                ),
            ]
        else:
            depth_metrics = []

        trainer = Trainer(
            "lidar_nerf",
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            optimizer=optimizer,
            criterion=criterion,
            ema_decay=0.95,
            fp16=opt.fp16,
            lr_scheduler=scheduler,
            scheduler_update_every_step=True,
            depth_metrics=depth_metrics,
            use_checkpoint=opt.ckpt,
            eval_interval=opt.eval_interval,
        )

        valid_loader = NeRFDataset(
            device=device,
            split="val",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            enable_lidar=opt.enable_lidar,
            num_rays_lidar=opt.num_rays_lidar,
        ).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        print(f"max_epoch: {max_epoch}")
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        test_loader = NeRFDataset(
            device=device,
            split="test",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            enable_lidar=opt.enable_lidar,
            num_rays_lidar=opt.num_rays_lidar,
        ).dataloader()

        if test_loader.has_gt:
            trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

        trainer.test(test_loader, write_video=True)  # test and save video

        trainer.save_mesh(resolution=128, threshold=10)


if __name__ == "__main__":
    main()
