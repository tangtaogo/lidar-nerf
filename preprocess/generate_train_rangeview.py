import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse

from lidarnerf.convert import (
    lidar_to_pano_with_intensities,
    lidar_to_pano_with_intensities_with_bbox_mask,
)


def all_points_to_world(pcd_path_list, lidar2world_list):
    pc_w_list = []
    for i, pcd_path in enumerate(pcd_path_list):
        point_cloud = np.load(pcd_path)
        point_cloud[:, -1] = 1
        points_world = (point_cloud @ (lidar2world_list[i].reshape(4, 4)).T)[:, :3]
        pc_w_list.append(points_world)
    return pc_w_list


def oriented_bounding_box(data):
    data_norm = data - data.mean(axis=0)
    C = np.cov(data_norm, rowvar=False)
    vals, vecs = np.linalg.eig(C)
    vecs = vecs[:, np.argsort(-vals)]
    Y = np.matmul(data_norm, vecs)
    offset = 0.03
    xmin = min(Y[:, 0]) - offset
    xmax = max(Y[:, 0]) + offset
    ymin = min(Y[:, 1]) - offset
    ymax = max(Y[:, 1]) + offset

    temp = list()
    temp.append([xmin, ymin])
    temp.append([xmax, ymin])
    temp.append([xmax, ymax])
    temp.append([xmin, ymax])

    pointInNewCor = np.asarray(temp)
    OBB = np.matmul(pointInNewCor, vecs.T) + data.mean(0)
    return OBB


def get_dataset_bbox(all_class, dataset_root, out_dir):
    object_bbox = {}
    for class_name in all_class:
        lidar_path = os.path.join(dataset_root, class_name)
        rt_path = os.path.join(lidar_path, "lidar2world.txt")
        filenames = os.listdir(lidar_path)
        filenames.remove("lidar2world.txt")
        filenames.sort(key=lambda x: int(x.split(".")[0]))
        show_interval = 1
        pcd_path_list = [os.path.join(lidar_path, filename) for filename in filenames][
            ::show_interval
        ]
        print(f"{lidar_path}: {len(pcd_path_list)} frames")
        lidar2world_list = list(np.loadtxt(rt_path))[::show_interval]
        all_points = all_points_to_world(pcd_path_list, lidar2world_list)
        pcd = np.concatenate(all_points).reshape((-1, 3))

        OBB_xy = oriented_bounding_box(pcd[:, :2])
        z_min, z_max = min(pcd[:, 2]), max(pcd[:, 2])
        OBB_buttum = np.concatenate([OBB_xy, np.tile(z_min, 4).reshape(4, 1)], axis=1)
        OBB_top = np.concatenate([OBB_xy, np.tile(z_max, 4).reshape(4, 1)], axis=1)
        OBB = np.concatenate([OBB_top, OBB_buttum])
        object_bbox[class_name] = OBB
    np.save(os.path.join(out_dir, "dataset_bbox_7k.npy"), object_bbox)


def LiDAR_2_Pano_NeRF_MVL(
    local_points_with_intensities,
    lidar_H,
    lidar_W,
    intrinsics,
    OBB_local,
    max_depth=80.0,
):
    pano, intensities = lidar_to_pano_with_intensities_with_bbox_mask(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        bbox_local=OBB_local,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view


def generate_nerf_mvl_train_data(
    H,
    W,
    intrinsics,
    all_class,
    dataset_bbox,
    nerf_mvl_parent_dir,
    out_dir,
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
    """

    for class_name in all_class:
        OBB = dataset_bbox[class_name]
        lidar_path = os.path.join(nerf_mvl_parent_dir, "nerf_mvl_7k", class_name)
        filenames = os.listdir(lidar_path)
        filenames.remove("lidar2world.txt")
        filenames.sort(key=lambda x: int(x.split(".")[0]))
        save_path = os.path.join(out_dir, class_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copy(
            os.path.join(lidar_path, "lidar2world.txt"),
            os.path.join(save_path, "lidar2world.txt"),
        )
        lidar2world = np.loadtxt(os.path.join(lidar_path, "lidar2world.txt"))
        avaliable_frames = [i for i in range(0, len(filenames))]
        print(class_name, " frames num ", len(avaliable_frames))
        for idx in tqdm(avaliable_frames):
            pcd = np.load(os.path.join(lidar_path, filenames[idx]))
            OBB_local = (
                np.concatenate([OBB, np.ones((8, 1))], axis=1)
                @ np.linalg.inv(lidar2world[idx].reshape(4, 4)).T
            )
            pano = LiDAR_2_Pano_NeRF_MVL(pcd, H, W, intrinsics, OBB_local)
            np.savez_compressed(
                os.path.join(save_path, "{:010d}.npz").format(idx), data=pano
            )


def create_nerf_mvl_rangeview():
    project_root = Path(__file__).parent.parent
    nerf_mvl_root = project_root / "data" / "nerf_mvl" / "nerf_mvl_7k"
    nerf_mvl_parent_dir = nerf_mvl_root.parent
    out_dir = nerf_mvl_parent_dir / "nerf_mvl_7k_pano"

    all_class = [
        "water_safety_barrier",
        "tire",
        "pier",
        "plant",
        "warning_sign",
        "traffic_cone",
        "bollard",
        "pedestrian",
        "car",
    ]

    # get_dataset_bbox
    if not os.path.exists(os.path.join(nerf_mvl_parent_dir, "dataset_bbox_7k.npy")):
        get_dataset_bbox(all_class, nerf_mvl_root, nerf_mvl_parent_dir)
    dataset_bbox = np.load(
        os.path.join(nerf_mvl_parent_dir, "dataset_bbox_7k.npy"), allow_pickle=True
    ).item()

    # generate train rangeview images
    H = 256
    W = 1800
    intrinsics = (15, 40)
    generate_nerf_mvl_train_data(
        H=H,
        W=W,
        intrinsics=intrinsics,
        all_class=all_class,
        dataset_bbox=dataset_bbox,
        nerf_mvl_parent_dir=nerf_mvl_parent_dir,
        out_dir=out_dir,
    )


def LiDAR_2_Pano_KITTI(
    local_points_with_intensities, lidar_H, lidar_W, intrinsics, max_depth=80.0
):
    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view


def generate_train_data(
    H,
    W,
    intrinsics,
    lidar_paths,
    out_dir,
    points_dim,
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lidar_path in tqdm(lidar_paths):
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        pano = LiDAR_2_Pano_KITTI(point_cloud, H, W, intrinsics)
        frame_name = lidar_path.split("/")[-1]
        suffix = frame_name.split(".")[-1]
        frame_name = frame_name.replace(suffix, "npy")
        np.save(out_dir / frame_name, pano)


def create_kitti_rangeview():
    project_root = Path(__file__).parent.parent
    kitti_360_root = project_root / "data" / "kitti360" / "KITTI-360"
    kitti_360_parent_dir = kitti_360_root.parent
    out_dir = kitti_360_parent_dir / "train"
    sequence_name = "2013_05_28_drive_0000"

    H = 66
    W = 1030
    intrinsics = (2.0, 26.9)  # fov_up, fov

    s_frame_id = 1908
    e_frame_id = 1971  # Inclusive
    frame_ids = list(range(s_frame_id, e_frame_id + 1))

    lidar_dir = (
        kitti_360_root
        / "data_3d_raw"
        / f"{sequence_name}_sync"
        / "velodyne_points"
        / "data"
    )
    lidar_paths = [
        os.path.join(lidar_dir, "%010d.bin" % frame_id) for frame_id in frame_ids
    ]

    generate_train_data(
        H=H,
        W=W,
        intrinsics=intrinsics,
        lidar_paths=lidar_paths,
        out_dir=out_dir,
        points_dim=4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti360",
        choices=["kitti360", "nerf_mvl"],
        help="The dataset loader to use.",
    )
    args = parser.parse_args()

    # Check dataset.
    if args.dataset == "kitti360":
        create_kitti_rangeview()
    elif args.dataset == "nerf_mvl":
        create_nerf_mvl_rangeview()


if __name__ == "__main__":
    main()
