import numpy as np

np.set_printoptions(suppress=True)
import os
import json
import tqdm
from lidarnerf.convert import pano_to_lidar


def cal_centerpose_bound_scale(
    lidar_rangeview_paths, lidar2worlds, intrinsics, bound=1.0
):
    near = 200
    far = 0
    points_world_list = []
    for i, lidar_rangeview_path in enumerate(lidar_rangeview_paths):
        pano = np.load(lidar_rangeview_path)
        point_cloud = pano_to_lidar(pano=pano[:, :, 2], lidar_K=intrinsics)
        point_cloud = np.concatenate(
            [point_cloud, np.ones(point_cloud.shape[0]).reshape(-1, 1)], -1
        )
        dis = np.sqrt(
            point_cloud[:, 0] ** 2 + point_cloud[:, 1] ** 2 + point_cloud[:, 2] ** 2
        )
        near = min(min(dis), near)
        far = max(far, max(dis))
        points_world = (point_cloud @ lidar2worlds[i].T)[:, :3]
        points_world_list.append(points_world)
    print("near, far:", near, far)

    # plt.figure(figsize=(16, 16))
    pc_all_w = np.concatenate(points_world_list)[:, :3]

    # plt.scatter(pc_all_w[:, 0], pc_all_w[:, 1], s=0.001)
    # lidar2world_scene = np.array(lidar2worlds)
    # plt.plot(lidar2world_scene[:, 0, -1], lidar2world_scene[:, 1, -1])
    # plt.savefig('vis/points-trajectory.png')

    centerpose = [
        (np.max(pc_all_w[:, 0]) + np.min(pc_all_w[:, 0])) / 2.0,
        (np.max(pc_all_w[:, 1]) + np.min(pc_all_w[:, 1])) / 2.0,
        (np.max(pc_all_w[:, 2]) + np.min(pc_all_w[:, 2])) / 2.0,
    ]
    print("centerpose: ", centerpose)
    pc_all_w_centered = pc_all_w - centerpose

    # plt.figure(figsize=(16, 16))
    # plt.scatter(pc_all_w_centered[:, 0], pc_all_w_centered[:, 1], s=0.001)
    # plt.savefig('vis/points-centered.png')

    bound_ori = [
        np.max(pc_all_w_centered[:, 0]),
        np.max(pc_all_w_centered[:, 1]),
        np.max(pc_all_w_centered[:, 2]),
    ]
    scale = bound / np.max(bound_ori)
    print("scale: ", scale)

    # pc_all_w_centered_scaled = pc_all_w_centered * scale
    # plt.figure(figsize=(16, 16))
    # plt.scatter(pc_all_w_centered_scaled[:, 0],
    #             pc_all_w_centered_scaled[:, 1],
    #             s=0.001)
    # plt.savefig('vis/points-centered-scaled.png')


def get_path_pose_from_json(root_path, sequence_id):
    with open(
        os.path.join(root_path, f"transforms_{sequence_id}_train.json"), "r"
    ) as f:
        transform = json.load(f)
    frames = transform["frames"]
    poses_lidar = []
    paths_lidar = []
    for f in tqdm.tqdm(frames, desc=f"Loading {type} data"):
        pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]
        f_lidar_path = os.path.join(root_path, f["lidar_file_path"])
        poses_lidar.append(pose_lidar)
        paths_lidar.append(f_lidar_path)
    return paths_lidar, poses_lidar


def main():
    # kitti360
    root_path = "data/kitti360"
    sequence_id = 1908
    lidar_rangeview_paths, lidar2worlds = get_path_pose_from_json(
        root_path, sequence_id=sequence_id
    )
    intrinsics = (2.0, 26.9)  # fov_up, fov

    cal_centerpose_bound_scale(lidar_rangeview_paths, lidar2worlds, intrinsics)


if __name__ == "__main__":
    main()
