import os
from nerfmvl_loader import NeRFMVLLoader
import numpy as np
import json
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    nerf_mvl_root = project_root / "data" / "nerf_mvl" / "nerf_mvl_7k_pano"
    nerf_mvl_parent_dir = nerf_mvl_root.parent

    # Specify frames and splits.
    train_split = {
        "water_safety_barrier": 2,
        "tire": 2,
        "pier": 2,
        "plant": 2,
        "warning_sign": 2,
        "bollard": 2,
        "pedestrian": 3,
        "car": 3,
        "traffic_cone": 3,
    }

    for class_name, split_intervel in train_split.items():
        # Get lidar paths (range view not raw data).
        range_view_dir = nerf_mvl_root / class_name
        filenames = os.listdir(range_view_dir)
        filenames.remove("lidar2world.txt")
        range_view_paths = [
            Path(os.path.join(range_view_dir, filename)) for filename in filenames
        ]
        num_samples = len(range_view_paths)
        frame_ids = np.arange(num_samples)

        train_frame_ids = [i for i in range(0, num_samples, split_intervel)]
        val_frame_ids = [i for i in range(0, num_samples, split_intervel * 20)]
        test_frame_ids = val_frame_ids

        # Load NeRF_MVL dataset.
        nerf_mvl_dataset = NeRFMVLLoader(nerf_mvl_root, class_name)

        # Get lidar2world.
        lidar2world = nerf_mvl_dataset.load_lidars(frame_ids)

        # Get image dimensions, assume all images have the same dimensions.
        lidar_range_image = np.load(range_view_paths[0])["data"]
        lidar_h, lidar_w, _ = lidar_range_image.shape

        # Split by train/test/val.
        all_indices = frame_ids
        train_indices = train_frame_ids
        val_indices = val_frame_ids
        test_indices = test_frame_ids

        split_to_all_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        for split, indices in split_to_all_indices.items():
            print(f"Split {split} has {len(indices)} frames.")

            lidar_paths_split = [range_view_paths[i] for i in indices]
            lidar2world_split = [lidar2world[i] for i in indices]
            json_dict = {
                "w_lidar": lidar_w,
                "h_lidar": lidar_h,
                "aabb_scale": 2,
                "frames": [
                    {
                        "lidar_file_path": str(
                            lidar_path.relative_to(nerf_mvl_parent_dir)
                        ),
                        "lidar2world": lidar2world.tolist(),
                    }
                    for (
                        lidar_path,
                        lidar2world,
                    ) in zip(
                        lidar_paths_split,
                        lidar2world_split,
                    )
                ],
            }
            json_path = nerf_mvl_parent_dir / f"transforms_{class_name}_{split}.json"

            with open(json_path, "w") as f:
                json.dump(json_dict, f, indent=2)
                print(f"Saved {json_path}.")


if __name__ == "__main__":
    main()
