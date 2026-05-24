from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Tuple

import tyro

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.zmq_core.camera_node import ZMQServerCamera

CAMERA_CONFIG = {
    "wrist": {"serial": "352122273671", "port": 5000},
    "top": {"serial": "105422061000", "port": 5001},
    "front": {"serial": "104122061227", "port": 5002},
}
CAMERA_ALIASES = {"writs": "wrist"}


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    camera_names: Tuple[str, ...] = ("wrist", "top", "front")
    interactive_select: bool = True
    width: int = 640
    height: int = 480
    fps: int = 30


def _canonical_camera_name(name: str) -> str:
    return CAMERA_ALIASES.get(name, name)


def _resolve_camera_names(camera_names: Tuple[str, ...]) -> List[str]:
    names = []
    for name in camera_names:
        canonical = _canonical_camera_name(name)
        if canonical not in CAMERA_CONFIG:
            valid = ", ".join(CAMERA_CONFIG)
            raise ValueError(f"Unknown camera `{name}`. Choose from: {valid}")
        if canonical not in names:
            names.append(canonical)
    return names


def _select_cameras_interactive(candidates: List[str]) -> List[str]:
    print("\nDetected configured cameras:")
    for i, name in enumerate(candidates):
        serial = CAMERA_CONFIG[name]["serial"]
        print(f"  [{i}] {name} ({serial})")
    print(
        "Choose cameras to record (comma-separated indices or names, Enter=all), e.g. `0,2` or `wrist,front`:"
    )
    user_input = input("> ").strip()
    if user_input == "":
        return candidates

    selected = []
    for token in user_input.split(","):
        token = token.strip()
        if token == "":
            continue
        if not token.isdigit():
            canonical = _canonical_camera_name(token)
            if canonical not in candidates:
                raise ValueError(f"Invalid camera name: `{token}`")
            selected.append(canonical)
            continue
        idx = int(token)
        if idx < 0 or idx >= len(candidates):
            raise ValueError(f"Camera index out of range: {idx}")
        selected.append(candidates[idx])

    deduped = []
    for serial in selected:
        if serial not in deduped:
            deduped.append(serial)
    return deduped


def launch_server(port: int, camera_id: str, camera_name: str, args: Args):
    camera = RealSenseCamera(
        camera_id, width=args.width, height=args.height, fps=args.fps
    )
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(
        f"Starting {camera_name} camera ({camera_id}) on "
        f"{args.width}x{args.height}@{args.fps}, port {port}"
    )
    server.serve()


def main(args):
    ids = get_device_ids()
    wanted_names = _resolve_camera_names(args.camera_names)
    available = [
        name for name in wanted_names if CAMERA_CONFIG[name]["serial"] in ids
    ]
    missing = [
        name for name in wanted_names if CAMERA_CONFIG[name]["serial"] not in ids
    ]

    if missing:
        print(f"Warning: configured cameras not found: {missing}")
    if len(available) == 0:
        raise RuntimeError(
            "None of the configured cameras are connected. "
            f"Configured: {wanted_names}, detected serials: {ids}"
        )

    selected: List[str] = available
    if args.interactive_select:
        selected = _select_cameras_interactive(available)
    if len(selected) == 0:
        raise RuntimeError("No cameras selected.")

    camera_servers = []
    for camera_name in selected:
        # start a python process for each camera
        camera_id = CAMERA_CONFIG[camera_name]["serial"]
        camera_port = CAMERA_CONFIG[camera_name]["port"]
        print(f"Launching {camera_name} camera ({camera_id})")
        camera_servers.append(
            Process(
                target=launch_server,
                args=(camera_port, camera_id, camera_name, args),
            )
        )

    for server in camera_servers:
        server.start()

    for server in camera_servers:
        server.join()


if __name__ == "__main__":
    main(tyro.cli(Args))
