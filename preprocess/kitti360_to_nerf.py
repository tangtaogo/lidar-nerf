from pathlib import Path

from kitti360_loader import KITTI360Loader
import camtools as ct
import numpy as np
import json


def normalize_Ts(Ts):
    # New Cs.
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])
    normalize_mat = ct.normalize.compute_normalize_mat(Cs)
    Cs_new = ct.project.homo_project(Cs.reshape((-1, 3)), normalize_mat)

    # New Ts.
    Ts_new = []
    for T, C_new in zip(Ts, Cs_new):
        pose = ct.convert.T_to_pose(T)
        pose[:3, 3] = C_new
        T_new = ct.convert.pose_to_T(pose)
        Ts_new.append(T_new)

    return Ts_new


def main():
    project_root = Path(__file__).parent.parent
    kitti_360_root = project_root / "data" / "kitti360" / "KITTI-360"
    kitti_360_parent_dir = kitti_360_root.parent

    # Specify frames and splits.
    sequence_name = "2013_05_28_drive_0000"
    sequence_id = "1908"

    if sequence_id == "1538":
        print("Using sqequence 1538-1601")
        s_frame_id = 1538
        e_frame_id = 1601  # Inclusive
        val_frame_ids = [1551, 1564, 1577, 1590]
    elif sequence_id == "1728":
        print("Using sqequence 1728-1791")
        s_frame_id = 1728
        e_frame_id = 1791  # Inclusive
        val_frame_ids = [1741, 1754, 1767, 1780]
    elif sequence_id == "1908":
        print("Using sqequence 1908-1971")
        s_frame_id = 1908
        e_frame_id = 1971  # Inclusive
        val_frame_ids = [1921, 1934, 1947, 1960]
    elif sequence_id == "3353":
        print("Using sqequence 3353-3416")
        s_frame_id = 3353
        e_frame_id = 3416  # Inclusive
        val_frame_ids = [3366, 3379, 3392, 3405]
    else:
        raise ValueError(f"Invalid sequence id: {sequence_id}")

    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    num_frames = len(frame_ids)

    test_frame_ids = val_frame_ids
    train_frame_ids = [x for x in frame_ids if x not in val_frame_ids]

    # Load KITTI-360 dataset.
    k3 = KITTI360Loader(kitti_360_root)

    # Get image paths.
    cam_00_im_paths = k3.get_image_paths("cam_00", sequence_name, frame_ids)
    cam_01_im_paths = k3.get_image_paths("cam_01", sequence_name, frame_ids)
    im_paths = cam_00_im_paths + cam_01_im_paths

    # Get Ks, Ts.
    cam_00_Ks, cam_00_Ts = k3.load_cameras("cam_00", sequence_name, frame_ids)
    cam_01_Ks, cam_01_Ts = k3.load_cameras("cam_01", sequence_name, frame_ids)
    Ks = np.concatenate([cam_00_Ks, cam_01_Ks], axis=0)
    Ts = np.concatenate([cam_00_Ts, cam_01_Ts], axis=0)
    # Ts = normalize_Ts(Ts)

    # Get image dimensions, assume all images have the same dimensions.
    im_rgb = ct.io.imread(cam_00_im_paths[0])
    im_h, im_w, _ = im_rgb.shape

    # Get lidar paths (range view not raw data).
    range_view_dir = kitti_360_parent_dir / "train"
    range_view_paths = [
        range_view_dir / "{:010d}.npy".format(int(frame_id)) for frame_id in frame_ids
    ]

    # Get lidar2world.
    lidar2world = k3.load_lidars(sequence_name, frame_ids)

    # Get image dimensions, assume all images have the same dimensions.
    lidar_range_image = np.load(range_view_paths[0])
    lidar_h, lidar_w, _ = lidar_range_image.shape

    # Split by train/test/val.
    all_indices = [i - s_frame_id for i in frame_ids]
    train_indices = [i - s_frame_id for i in train_frame_ids]
    val_indices = [i - s_frame_id for i in val_frame_ids]
    test_indices = [i - s_frame_id for i in test_frame_ids]

    # all_indices = all_indices + [i + num_frames for i in all_indices]
    # train_indices = train_indices + [i + num_frames for i in train_indices]
    # val_indices = val_indices + [i + num_frames for i in val_indices]
    # test_indices = test_indices + [i + num_frames for i in test_indices]

    split_to_all_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    for split, indices in split_to_all_indices.items():
        print(f"Split {split} has {len(indices)} frames.")
        im_paths_split = [im_paths[i] for i in indices]
        lidar_paths_split = [range_view_paths[i] for i in indices]
        lidar2world_split = [lidar2world[i] for i in indices]
        Ks_split = [Ks[i] for i in indices]
        Ts_split = [Ts[i] for i in indices]

        json_dict = {
            "w": im_w,
            "h": im_h,
            "w_lidar": lidar_w,
            "h_lidar": lidar_h,
            "fl_x": Ks_split[0][0, 0],
            "fl_y": Ks_split[0][1, 1],
            "cx": Ks_split[0][0, 2],
            "cy": Ks_split[0][1, 2],
            "aabb_scale": 2,
            "frames": [
                {
                    "file_path": str(path.relative_to(kitti_360_parent_dir)),
                    "transform_matrix": ct.convert.T_to_pose(T).tolist(),
                    "lidar_file_path": str(
                        lidar_path.relative_to(kitti_360_parent_dir)
                    ),
                    "lidar2world": lidar2world.tolist(),
                }
                for (
                    path,
                    T,
                    lidar_path,
                    lidar2world,
                ) in zip(
                    im_paths_split,
                    Ts_split,
                    lidar_paths_split,
                    lidar2world_split,
                )
            ],
        }
        json_path = kitti_360_parent_dir / f"transforms_{sequence_id}_{split}.json"

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            print(f"Saved {json_path}.")


if __name__ == "__main__":
    main()
