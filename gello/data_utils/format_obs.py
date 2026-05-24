import datetime
from pathlib import Path
from typing import Dict

import numpy as np


class HDF5EpisodeWriter:
    """HDF5 writer matching the scripted RealSense episode schema."""

    def __init__(
        self,
        folder: Path,
        timestamp: datetime.datetime,
    ) -> None:
        import h5py

        folder.mkdir(exist_ok=True, parents=True)
        self.file_path = folder / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.h5"
        self._h5 = h5py.File(self.file_path, "w")
        self._h5.attrs["sim"] = False
        self._obs_group = self._h5.create_group("observations")
        self._image_group = self._obs_group.create_group("images")
        self._datasets: Dict[str, object] = {}
        self._length = 0

    def _append_dataset(self, path: str, value: np.ndarray, compression=None) -> None:
        array = np.asarray(value)
        if path not in self._datasets:
            if "/" in path:
                group_path, name = path.rsplit("/", 1)
                group = self._h5[group_path]
            else:
                group = self._h5
                name = path
            self._datasets[path] = group.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                dtype=array.dtype,
                chunks=True,
                compression=compression,
            )
        dataset = self._datasets[path]
        dataset.resize((self._length + 1, *array.shape))
        dataset[self._length] = array

    def _get_rgb(self, obs: Dict[str, np.ndarray], camera_name: str) -> np.ndarray:
        image = obs.get(f"{camera_name}_rgb")
        if image is not None:
            return image
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def append(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> None:
        qpos = np.asarray(obs["joint_positions"])[:6]
        gripper_state = np.asarray(obs["gripper_position"]).reshape(-1)[0]
        action_cmd = np.asarray(action)[:6]
        timestamp = np.array(datetime.datetime.now().timestamp(), dtype=np.float64)

        self._append_dataset(
            "observations/images/cam1_rgb",
            self._get_rgb(obs, "wrist"),
            compression="lzf",
        )
        self._append_dataset(
            "observations/images/cam2_rgb",
            self._get_rgb(obs, "top"),
            compression="lzf",
        )
        self._append_dataset(
            "observations/images/cam3_rgb",
            self._get_rgb(obs, "front"),
            compression="lzf",
        )
        self._append_dataset(
            "observations/images/aria_rgb",
            np.zeros((480, 640, 3), dtype=np.uint8),
            compression="lzf",
        )
        self._append_dataset("observations/qpos", qpos)
        self._append_dataset("observations/gripper_state", np.array(gripper_state))
        self._append_dataset("action", action_cmd)
        self._append_dataset("timestamp", timestamp)
        self._length += 1

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.attrs["num_frames"] = self._length
            self._h5.close()
            self._h5 = None
