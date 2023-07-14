from pathlib import Path
import numpy as np
import camtools as ct
import open3d as o3d


class KITTI360Loader:
    def __init__(self, kitti_360_root) -> None:
        # Root directory.
        self.kitti_360_root = Path(kitti_360_root)
        if not self.kitti_360_root.is_dir():
            raise FileNotFoundError(f"KITTI-360 {kitti_360_root} not found.")

        # Other directories.
        self.calibration_dir = self.kitti_360_root / "calibration"
        self.data_poses_dir = self.kitti_360_root / "data_poses"
        self.data_2d_raw_dir = self.kitti_360_root / "data_2d_raw"
        self.data_3d_raw_dir = self.kitti_360_root / "data_3d_raw"

        # Check if all directories exist.
        if not self.calibration_dir.is_dir():
            raise FileNotFoundError(
                f"Calibration dir {self.calibration_dir} not found."
            )
        if not self.data_poses_dir.is_dir():
            raise FileNotFoundError(f"Data poses dir {self.data_poses_dir} not found.")
        if not self.data_2d_raw_dir.is_dir():
            raise FileNotFoundError(
                f"Data 2D raw dir {self.data_2d_raw_dir} not found."
            )
        if not self.data_3d_raw_dir.is_dir():
            raise FileNotFoundError(
                f"Data 3D raw dir {self.data_3d_raw_dir} not found."
            )

    @staticmethod
    def _read_variable(fid, name, M, N):
        """
        Ref:
            kitti360scripts/devkits/commons/loadCalibration.py
        """
        # Rewind
        fid.seek(0, 0)

        # Search for variable identifier
        line = 1
        success = 0
        while line:
            line = fid.readline()
            if line.startswith(name):
                success = 1
                break

        # Return if variable identifier not found
        if success == 0:
            return None

        # Fill matrix
        line = line.replace("%s:" % name, "")
        line = line.split()
        assert len(line) == M * N
        line = [float(x) for x in line]
        mat = np.array(line).reshape(M, N)

        return mat

    @staticmethod
    def _load_perspective_intrinsics(intrinsics_path):
        """
        Args:
            intrinsics_path: str, path to perspective.txt.

        Returns:
            A dict, containing:
            - "P_rect_00": 4x4 rectified intrinsic for cam_00.
            - "P_rect_01": 4x4 rectified intrinsic for cam_01.
            - "R_rect_00": 3x3 rectification matrix for cam_00.
            - "R_rect_01": 3xe rectification matrix for cam_01.

        Ref:
            kitti360scripts/devkits/commons/loadCalibration.py::loadPerspectiveIntrinsic
        """
        intrinsics_path = Path(intrinsics_path)
        with open(intrinsics_path, "r") as fid:
            perspective_dict = {}
            intrinsic_names = ["P_rect_00", "R_rect_00", "P_rect_01", "R_rect_01"]
            last_row = np.array([0, 0, 0, 1]).reshape(1, 4)
            for intrinsic in intrinsic_names:
                if intrinsic.startswith("P_rect"):
                    perspective_dict[intrinsic] = np.concatenate(
                        (KITTI360Loader._read_variable(fid, intrinsic, 3, 4), last_row)
                    )
                else:
                    perspective_dict[intrinsic] = KITTI360Loader._read_variable(
                        fid, intrinsic, 3, 3
                    )
        return perspective_dict

    def load_images(self, camera_name, sequence_name, frame_ids):
        """
        Args:
            camera_name: str, name of camera. e.g. "cam_00".
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".
            frame_ids: list of int, frame ids. e.g. range(1908, 1971+1).

        Returns:
            An np.ndarray, float32, [N, H, W, 3], range 0-1, RGB images.
        """
        im_paths = self.get_image_paths(camera_name, sequence_name, frame_ids)
        ims = [ct.io.imread(im_path) for im_path in im_paths]
        ims = np.stack(ims, axis=0)

        return ims

    def get_image_paths(self, camera_name, sequence_name, frame_ids):
        """
        Args:
            camera_name: str, name of camera. e.g. "cam_00".
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".
            frame_ids: list of int, frame ids. e.g. range(1908, 1971+1).

        Returns:
            An list of str, image paths.
        """
        # Sanity checks.
        if camera_name == "cam_00":
            subdir_name = "image_00"
        elif camera_name == "cam_01":
            subdir_name = "image_01"
        else:
            raise ValueError(f"Invalid camera_name {camera_name}")

        # Get image paths.
        im_dir = (
            self.data_2d_raw_dir / f"{sequence_name}_sync" / subdir_name / "data_rect"
        )
        im_paths = [im_dir / f"{frame_id:010d}.png" for frame_id in frame_ids]
        for im_path in im_paths:
            if not im_path.is_file():
                raise FileNotFoundError(f"Image {im_path} not found.")

        return im_paths

    def _load_all_cameras(self, sequence_name):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".

        Returns:
            cam_00_K: 3x3 intrinsics, rectified perspective cam_00.
            cam_01_K: 3x3 intrinsics, rectified perspective cam_01.
            cam_00_T_dict: map frame_id to 4x4 T, rectified perspective cam_00.
            cam_01_T_dict: map frame_id to 4x4 T, rectified perspective cam_01.
        """
        data_poses_dir = self.data_poses_dir / f"{sequence_name}_sync"
        assert data_poses_dir.is_dir()

        # Load intrinsics and rectification matrices.
        perspective_path = self.calibration_dir / "perspective.txt"
        perspective_dict = KITTI360Loader._load_perspective_intrinsics(perspective_path)
        cam_00_K = perspective_dict["P_rect_00"][:3, :3]  # 3x3
        cam_01_K = perspective_dict["P_rect_01"][:3, :3]  # 3x3
        cam_00_rec = np.eye(4)  # 4x4
        cam_00_rec[:3, :3] = perspective_dict["R_rect_00"]
        cam_01_rec = np.eye(4)  # 4x4
        cam_01_rec[:3, :3] = perspective_dict["R_rect_01"]

        # IMU to world transformation (poses.txt).
        poses_path = data_poses_dir / "poses.txt"
        imu_to_world_dict = dict()
        frame_ids = []
        for line in np.loadtxt(poses_path):
            frame_id = int(line[0])
            frame_ids.append(frame_id)
            imu_to_world = line[1:].reshape((3, 4))
            imu_to_world_dict[frame_id] = imu_to_world

        # Camera to IMU transformation (calib_cam_to_pose.txt).
        cam_to_imu_path = self.calibration_dir / "calib_cam_to_pose.txt"
        with open(cam_to_imu_path, "r") as fid:
            cam_00_to_imu = KITTI360Loader._read_variable(fid, "image_00", 3, 4)
            cam_01_to_imu = KITTI360Loader._read_variable(fid, "image_01", 3, 4)
            cam_02_to_imu = KITTI360Loader._read_variable(fid, "image_02", 3, 4)
            cam_03_to_imu = KITTI360Loader._read_variable(fid, "image_03", 3, 4)
            cam_00_to_imu = ct.convert.pad_0001(cam_00_to_imu)
            cam_01_to_imu = ct.convert.pad_0001(cam_01_to_imu)
            cam_02_to_imu = ct.convert.pad_0001(cam_02_to_imu)
            cam_03_to_imu = ct.convert.pad_0001(cam_03_to_imu)

        # Compute rectified cam_00_to_world, cam_01_to_world.
        cam_00_to_world_dict = dict()
        for frame_id in frame_ids:
            imu_to_world = imu_to_world_dict[frame_id]
            cam_00_to_world_unrec = imu_to_world @ cam_00_to_imu
            cam_00_to_world = cam_00_to_world_unrec @ np.linalg.inv(cam_00_rec)
            cam_00_to_world_dict[frame_id] = ct.convert.pad_0001(cam_00_to_world)
        cam_01_to_world_dict = dict()
        for frame_id in frame_ids:
            imu_to_world = imu_to_world_dict[frame_id]
            cam_01_to_world_unrec = imu_to_world @ cam_01_to_imu
            cam_01_to_world = cam_01_to_world_unrec @ np.linalg.inv(cam_01_rec)
            cam_01_to_world_dict[frame_id] = ct.convert.pad_0001(cam_01_to_world)

        # Sanity check: check our rectified cam0_to_world is the same as the
        # ones ground-truth given by KITTI-360.
        cam_00_to_world_path = data_poses_dir / "cam0_to_world.txt"
        gt_cam_00_to_world_dict = dict()
        for line in np.loadtxt(cam_00_to_world_path):
            frame_id = int(line[0])
            gt_cam_00_to_world_dict[frame_id] = line[1:].reshape((4, 4))
        for frame_id in frame_ids:
            gt_cam_00_to_world = gt_cam_00_to_world_dict[frame_id]
            cam_00_to_world = cam_00_to_world_dict[frame_id]
            assert np.allclose(
                gt_cam_00_to_world, cam_00_to_world, atol=1e-5, rtol=1e-5
            )

        # Convert cam_to_world to T.
        cam_00_T_dict = dict()
        cam_01_T_dict = dict()
        for frame_id in frame_ids:
            cam_00_T = np.linalg.inv(cam_00_to_world_dict[frame_id])
            cam_01_T = np.linalg.inv(cam_01_to_world_dict[frame_id])
            cam_00_T_dict[frame_id] = cam_00_T
            cam_01_T_dict[frame_id] = cam_01_T

        return cam_00_K, cam_01_K, cam_00_T_dict, cam_01_T_dict

    def load_cameras(self, camera_name, sequence_name, frame_ids):
        """
        Args:
            camera_name: str, name of camera. e.g. "cam_00".
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".
            frame_ids: list of int, frame ids. e.g. range(1908, 1971+1).

        Returns:
            Ks, Ts
        """
        (
            cam_00_K,
            cam_01_K,
            cam_00_T_dict,
            cam_01_T_dict,
        ) = self._load_all_cameras(sequence_name)
        num_cameras = len(frame_ids)

        if camera_name == "cam_00":
            Ks = [cam_00_K for _ in range(num_cameras)]
            Ts = [cam_00_T_dict[frame_id] for frame_id in frame_ids]
        elif camera_name == "cam_01":
            Ks = [cam_01_K for _ in range(num_cameras)]
            Ts = [cam_01_T_dict[frame_id] for frame_id in frame_ids]
        else:
            raise ValueError(f"Unknown camera name {camera_name}")

        Ks = np.stack(Ks)
        Ts = np.stack(Ts)
        return Ks, Ts

    def _load_all_lidars(self, sequence_name):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".

        Returns:
            velo_to_world: 4x4 metric.
        """
        data_poses_dir = self.data_poses_dir / f"{sequence_name}_sync"
        assert data_poses_dir.is_dir()

        # IMU to world transformation (poses.txt).
        poses_path = data_poses_dir / "poses.txt"
        imu_to_world_dict = dict()
        frame_ids = []
        for line in np.loadtxt(poses_path):
            frame_id = int(line[0])
            frame_ids.append(frame_id)
            imu_to_world = line[1:].reshape((3, 4))
            imu_to_world_dict[frame_id] = imu_to_world

        # Camera to IMU transformation (calib_cam_to_pose.txt).
        cam_to_imu_path = self.calibration_dir / "calib_cam_to_pose.txt"
        with open(cam_to_imu_path, "r") as fid:
            cam_00_to_imu = KITTI360Loader._read_variable(fid, "image_00", 3, 4)
            cam_00_to_imu = ct.convert.pad_0001(cam_00_to_imu)

        # Camera00 to Velo transformation (calib_cam_to_velo.txt).
        cam00_to_velo_path = self.calibration_dir / "calib_cam_to_velo.txt"
        with open(cam00_to_velo_path, "r") as fid:
            line = fid.readline().split()
            line = [float(x) for x in line]
            cam_00_to_velo = np.array(line).reshape(3, 4)
            cam_00_to_velo = ct.convert.pad_0001(cam_00_to_velo)

        # Compute velo_to_world
        velo_to_world_dict = dict()
        for frame_id in frame_ids:
            imu_to_world = imu_to_world_dict[frame_id]
            cam_00_to_world_unrec = imu_to_world @ cam_00_to_imu
            velo_to_world = cam_00_to_world_unrec @ np.linalg.inv(cam_00_to_velo)
            velo_to_world_dict[frame_id] = ct.convert.pad_0001(velo_to_world)

        return velo_to_world_dict

    def load_lidars(self, sequence_name, frame_ids):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".
            frame_ids: list of int, frame ids. e.g. range(1908, 1971+1).

        Returns:
            velo_to_worlds
        """
        velo_to_world_dict = self._load_all_lidars(sequence_name)
        velo_to_worlds = [velo_to_world_dict[frame_id] for frame_id in frame_ids]
        velo_to_worlds = np.stack(velo_to_worlds)
        return velo_to_worlds


def main():
    # Load cameras.
    k3 = KITTI360Loader(kitti_360_root=Path("data") / "KITTI-360")
    cam_00_Ks, cam_00_Ts = k3.load_cameras(
        camera_name="cam_00",
        sequence_name="2013_05_28_drive_0000",
        frame_ids=range(1908, 1971 + 1),
    )
    cam_01_Ks, cam_01_Ts = k3.load_cameras(
        camera_name="cam_01",
        sequence_name="2013_05_28_drive_0000",
        frame_ids=range(1908, 1971 + 1),
    )

    # Load images.
    im_cam_00s = k3.load_images(
        camera_name="cam_00",
        sequence_name="2013_05_28_drive_0000",
        frame_ids=range(1908, 1971 + 1),
    )
    im_cam_01s = k3.load_images(
        camera_name="cam_01",
        sequence_name="2013_05_28_drive_0000",
        frame_ids=range(1908, 1971 + 1),
    )

    # Visualize.
    cam_00_frames = ct.camera.create_camera_ray_frames(cam_00_Ks, cam_00_Ts, size=0.8)
    cam_01_frames = ct.camera.create_camera_ray_frames(cam_01_Ks, cam_01_Ts, size=0.8)
    o3d.visualization.draw_geometries([cam_00_frames, cam_01_frames])


if __name__ == "__main__":
    main()
