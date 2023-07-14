from pathlib import Path
import numpy as np


class NeRFMVLLoader:
    def __init__(self, nerf_mvl_root, class_name) -> None:
        # Root directory.
        self.nerf_mvl_root = Path(nerf_mvl_root)
        if not self.nerf_mvl_root.is_dir():
            raise FileNotFoundError(f"NeRF_MVL {nerf_mvl_root} not found.")

        # Other directories.
        self.data_3d_raw_dir = self.nerf_mvl_root / class_name
        self.lidar2world_path = self.data_3d_raw_dir / "lidar2world.txt"

        # Check if all directories exist.
        if not self.data_3d_raw_dir.is_dir():
            raise FileNotFoundError(
                f"Data 3D raw dir {self.data_3d_raw_dir} not found."
            )

    def _load_all_lidars(
        self,
    ):
        """
        Args:

        Returns:
            velo_to_world: 4x4 metric.
        """

        velo_to_world_dict = np.loadtxt(self.lidar2world_path)
        return velo_to_world_dict.reshape(-1, 4, 4)

    def load_lidars(self, frame_ids):
        """
        Args:
            frame_ids: list of int, frame ids. e.g. range(1908, 1971+1).

        Returns:
            velo_to_worlds
        """
        velo_to_world_dict = self._load_all_lidars()
        velo_to_worlds = [velo_to_world_dict[frame_id] for frame_id in frame_ids]
        velo_to_worlds = np.stack(velo_to_worlds)
        return velo_to_worlds


def main():
    dataset = NeRFMVLLoader(Path("data") / "nerf_mvl" / "nerf_mvl_7k_pano", "pier")
    velo_to_world_dict = dataset._load_all_lidars()
    return


if __name__ == "__main__":
    main()
