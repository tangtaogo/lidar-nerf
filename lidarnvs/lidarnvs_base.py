from abc import ABC, abstractmethod

import numpy as np


class LidarNVSBase(ABC):
    @abstractmethod
    def fit(self, dataset) -> None:
        """
        Fit the model to the given train dataset.

        Args:
            dataset: A NeRFDataset object.
        """

    @abstractmethod
    def predict_frame(
        self,
        lidar_K: np.ndarray,  # (2, )
        lidar_pose: np.ndarray,  # (4, 4)
        lidar_H: int,
        lidar_W: int,
    ) -> dict:
        """
        Predict (synthesis) the point cloud from the given lidar parameters.
        All necessary information parameters to model a lidar are given.

        Args:
            lidar_K: (2, ), float32
            lidar_pose: (4, 4), float32
            lidar_H: int
            lidar_W: int

        Return:
            predict_dict: dict
            - ["local_points"]: (N, 3), float32
            - ["points"]      : (N, 3), float32
            - ["pano"]        : (H, W), float32
            - ["intensities"] : (H, W), float32
        """

    @abstractmethod
    def predict_frame_with_raydrop(
        self,
        lidar_K: np.ndarray,  # (2, )
        lidar_pose: np.ndarray,  # (4, 4)
        lidar_H: int,
        lidar_W: int,
    ) -> dict:
        pass
