from pathlib import Path

import numpy as np

from lidarnvs.lidarnvs_pcgen import LidarNVSPCGen, generate_raydrop_data_pcgen
from lidarnvs.lidarnvs_poisson import LidarNVSPoisson
from lidarnvs.lidarnvs_nksr import LidarNVSNksr
from lidarnvs.lidarnvs_meshing import generate_raydrop_data_meshing
from lidarnvs.loader import extract_dataset_frame
from lidarnvs.eval import eval_points_and_pano
from tqdm import tqdm
import pickle
import argparse
from lidarnerf.dataset.kitti360_dataset import KITTI360Dataset
from lidarnerf.dataset.nerfmvl_dataset import NeRFMVLDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti360",
        choices=["kitti360", "nerf_mvl"],
        help="The dataset loader to use.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="poisson",
        choices=["poisson", "nksr", "pcgen"],
        help="method for lidarnvs",
    )
    parser.add_argument(
        "--raycasting",
        type=str,
        default="cp",
        choices=["cp", "fpa"],
        help="raycasting mehtod",
    )
    # dataset
    parser.add_argument("--path", type=str, default="data/kitti360")
    parser.add_argument(
        "--sequence_id",
        type=str,
        default="1908",
        help="The sequence id within the selected dataset to use.",
    )
    parser.add_argument(
        "--num_rays_lidar",
        type=int,
        default=4096,
        help="num rays sampled per image for each training step",
    )
    parser.add_argument(
        "--offset", type=float, nargs="*", default=[0, 0, 0], help="offset of location"
    )
    parser.add_argument(
        "--enable_collect_raydrop_dataset",
        action="store_true",
        help="Whether to collect raydrop dataset. If not enabled, inference "
        "mode will be used",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="The ckpt of raydrop network.",
    )
    parser.add_argument(
        "--poisson_depth",
        type=int,
        default=11,
        help="Depth of tree for Poisson recon.",
    )
    parser.add_argument(
        "--poisson_min_density",
        type=float,
        default=0.3,
        help="Minimum density to filter points after Poisson recon.",
    )
    args = parser.parse_args()

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
    if args.dataset == "kitti360":
        if args.sequence_id not in kitti360_sequence_ids:
            raise ValueError(
                f"Unknown sequence id {args.sequence_id} for {args.dataset}"
            )
    elif args.dataset == "nerf_mvl":
        if args.sequence_id not in nerf_mvl_sequence_ids:
            raise ValueError(
                f"Unknown sequence id {args.sequence_id} for {args.dataset}"
            )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("[Config]===============================================")
    print(f"dataset             : {args.dataset}")
    print(f"sequence_id         : {args.sequence_id}")
    print(f"poisson_depth       : {args.poisson_depth}")
    print(f"poisson_min_density : {args.poisson_min_density}")
    print(f"dataset_collect_mode: {args.enable_collect_raydrop_dataset}")
    print("=======================================================")

    # Init train and test datasets.
    if args.dataset == "kitti360":
        train_dataset = KITTI360Dataset(
            split="train",
            root_path=args.path,
            offset=args.offset,
            num_rays_lidar=args.num_rays_lidar,
            sequence_id=args.sequence_id,
        )

        test_dataset = KITTI360Dataset(
            split="train",
            root_path=args.path,
            offset=args.offset,
            num_rays_lidar=args.num_rays_lidar,
            sequence_id=args.sequence_id,
        )
    elif args.dataset == "nerf_mvl":
        train_dataset = NeRFMVLDataset(
            split="train",
            root_path=args.path,
            offset=args.offset,
            num_rays_lidar=args.num_rays_lidar,
            sequence_id=args.sequence_id,
        )
        test_dataset = NeRFMVLDataset(
            split="test",
            root_path=args.path,
            offset=args.offset,
            num_rays_lidar=args.num_rays_lidar,
            sequence_id=args.sequence_id,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Train.
    if args.enable_collect_raydrop_dataset:
        ckpt_path = None
    else:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.is_file():
            raise ValueError(f"ckpt_path ({ckpt_path}) does not exist.")

    if args.method == "poisson":
        nvs = LidarNVSPoisson(
            poisson_depth=args.poisson_depth,
            poisson_min_density=args.poisson_min_density,
            intensity_interpolate_k=9,
            ckpt_path=ckpt_path,
        )
    elif args.method == "nksr":
        nvs = LidarNVSNksr(ckpt_path=ckpt_path)
    elif args.method == "pcgen":
        nvs = LidarNVSPCGen(
            raycasting=args.raycasting,
            ckpt_path=ckpt_path,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    nvs.fit(train_dataset)
    exit(0)

    # Eval test frames.
    all_metrics = []
    for frame_idx in tqdm(range(len(test_dataset)), desc="Eval test frames"):
        gt_frame = extract_dataset_frame(test_dataset, frame_idx=frame_idx)
        if args.enable_collect_raydrop_dataset:
            inference_func = nvs.predict_frame
        else:
            inference_func = nvs.predict_frame_with_raydrop
        pd_frame = inference_func(
            lidar_K=gt_frame["lidar_K"],
            lidar_pose=gt_frame["lidar_pose"],
            lidar_H=gt_frame["lidar_H"],
            lidar_W=gt_frame["lidar_W"],
        )
        if args.dataset == "nerf_mvl":
            # Load values to be updated.
            gt_intensities = gt_frame["intensities"]
            pd_intensities = pd_frame["intensities"]
            gt_pano = gt_frame["pano"]
            pd_pano = pd_frame["pano"]

            # Load mask.
            pano_mask = gt_frame["pano_mask"]
            nonzero_idx = np.array(np.nonzero(pano_mask))
            new_h = max(nonzero_idx[0]) - min(nonzero_idx[0]) + 1
            new_w = max(nonzero_idx[1]) - min(nonzero_idx[1]) + 1
            gt_intensities = gt_intensities[pano_mask].reshape(new_h, new_w)
            pd_intensities = pd_intensities[pano_mask].reshape(new_h, new_w)
            gt_intensities = gt_intensities * 255
            pd_intensities = pd_intensities * 255
            gt_pano = gt_pano[pano_mask].reshape(new_h, new_w)
            pd_pano = pd_pano[pano_mask].reshape(new_h, new_w)

            metrics = eval_points_and_pano(
                gt_local_points=gt_frame["local_points"],
                pd_local_points=pd_frame["local_points"],
                gt_intensities=gt_intensities,
                pd_intensities=pd_intensities,
                gt_pano=gt_pano,
                pd_pano=pd_pano,
            )
        else:
            metrics = eval_points_and_pano(
                gt_local_points=gt_frame["local_points"],
                pd_local_points=pd_frame["local_points"],
                gt_intensities=gt_frame["intensities"],
                pd_intensities=pd_frame["intensities"],
                gt_pano=gt_frame["pano"],
                pd_pano=pd_frame["pano"],
            )
        all_metrics.append(metrics)

    # Compute mean metrics.
    mean_metrics = {}
    for key in all_metrics[0].keys():
        mean_metrics[key] = np.mean([m[key] for m in all_metrics])
    print("Mean metrics:")
    for key in sorted(mean_metrics.keys()):
        print(f"- {key}: {mean_metrics[key]:.4f}")

    # # Visualize a single test frame.
    # gt_pcd = o3d.geometry.PointCloud()
    # gt_pcd.points = o3d.utility.Vector3dVector(gt_frame["points"])
    # gt_pcd.colors = o3d.utility.Vector3dVector(
    #     ct.colormap.query(gt_frame["point_intensities"]))

    # pd_pcd = o3d.geometry.PointCloud()
    # pd_pcd.points = o3d.utility.Vector3dVector(train_frame_nvs["points"])
    # pd_pcd.colors = o3d.utility.Vector3dVector(
    #     ct.colormap.query(train_frame_nvs["point_intensities"]))

    # o3d.visualization.draw_geometries([gt_pcd])
    # o3d.visualization.draw_geometries([pd_pcd])

    # Concat all in to big tensors.
    if args.enable_collect_raydrop_dataset:
        if args.method == "poisson" and args.dataset != "nerf_mvl":
            raydrop_train_data = generate_raydrop_data_meshing(train_dataset, nvs)
            raydrop_test_data = generate_raydrop_data_meshing(test_dataset, nvs)
        elif args.method == "pcgen":
            raydrop_train_data = generate_raydrop_data_pcgen(
                train_dataset, nvs, rm_pano_mask=False
            )
            raydrop_test_data = generate_raydrop_data_pcgen(
                test_dataset, nvs, rm_pano_mask=False
            )
        else:
            raise ValueError(f"Unknown method/dataset: {args.method}/{args.dataset}")

        data_dir = (
            Path("data/raydrop") / args.method / f"{args.dataset}_{args.sequence_id}"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        train_data_path = data_dir / "train_data.pkl"
        test_data_path = data_dir / "test_data.pkl"

        with open(train_data_path, "wb") as f:
            pickle.dump(raydrop_train_data, f)
        with open(test_data_path, "wb") as f:
            pickle.dump(raydrop_test_data, f)


if __name__ == "__main__":
    main()
