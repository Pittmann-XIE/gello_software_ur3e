from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tyro

from gello.zmq_core.camera_node import ZMQClientCamera

CAMERA_NAME_TO_PORT = {
    "wrist": 5000,
    "top": 5001,
    "front": 5002,
}
CAMERA_ALIASES = {"writs": "wrist"}


@dataclass
class Args:
    camera_names: Tuple[str, ...] = ("wrist", "top", "front")
    hostname: str = "127.0.0.1"
    # hostname: str = "128.32.175.167"


def _canonical_camera_name(name: str) -> str:
    return CAMERA_ALIASES.get(name, name)


def main(args):
    cameras = []
    import cv2

    images_display_names = []
    for camera_name in args.camera_names:
        camera_name = _canonical_camera_name(camera_name)
        if camera_name not in CAMERA_NAME_TO_PORT:
            valid = ", ".join(CAMERA_NAME_TO_PORT)
            raise ValueError(f"Unknown camera `{camera_name}`. Choose from: {valid}")
        port = CAMERA_NAME_TO_PORT[camera_name]
        cameras.append(ZMQClientCamera(port=port, host=args.hostname))
        images_display_names.append(camera_name)
        cv2.namedWindow(images_display_names[-1], cv2.WINDOW_NORMAL)

    while True:
        for display_name, camera in zip(images_display_names, cameras):
            image, depth = camera.read()
            stacked_depth = np.dstack([depth, depth, depth]).astype(np.uint8)
            image_depth = cv2.hconcat([image[:, :, ::-1], stacked_depth])
            cv2.imshow(display_name, image_depth)
            cv2.waitKey(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
