import atexit
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False
CAMERA_NAME_TO_PORT = {
    "wrist": 5000,
    "top": 5001,
    "front": 5002,
}
CAMERA_ALIASES = {"writs": "wrist"}


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    use_save_interface: bool = False
    """Enable saving data with keyboard interface."""
    use_camera_clients: bool = False
    """Attach ZMQ camera clients to observations for recording."""
    camera_names: Tuple[str, ...] = ("wrist", "top", "front")
    """Camera names to attach when use_camera_clients=True."""
    camera_host: str = "127.0.0.1"
    """Camera server host for ZMQ camera clients."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def validate_leader_follower_dims(env, agent):
    """Validate that leader and follower have matching action/state dimensions."""
    obs = env.get_obs()
    leader_action = agent.act(obs)
    follower_joints = obs["joint_positions"]
    if len(leader_action) != len(follower_joints):
        raise ValueError(
            "Leader/follower dimension mismatch: "
            f"agent outputs {len(leader_action)} DOF, "
            f"robot reports {len(follower_joints)} DOF. "
            "Check UR gripper mode (`robot.no_gripper`) and agent start/config."
        )


def _canonical_camera_name(name: str) -> str:
    return CAMERA_ALIASES.get(name, name)


def resolve_camera_names(camera_names: Tuple[str, ...]) -> Tuple[str, ...]:
    resolved = []
    for name in camera_names:
        canonical = _canonical_camera_name(name)
        if canonical not in CAMERA_NAME_TO_PORT:
            valid = ", ".join(CAMERA_NAME_TO_PORT)
            raise ValueError(f"Unknown camera `{name}`. Choose from: {valid}")
        if canonical not in resolved:
            resolved.append(canonical)
    return tuple(resolved)


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))
    if args.use_camera_clients:
        from gello.zmq_core.camera_node import ZMQClientCamera

        camera_names = resolve_camera_names(args.camera_names)
        camera_dict = {
            name: ZMQClientCamera(
                port=CAMERA_NAME_TO_PORT[name], host=args.camera_host
            )
            for name in camera_names
        }
        env = RobotEnv(
            robot_client, control_rate_hz=cfg.get("hz", 30), camera_dict=camera_dict
        )
        print(f"Attached camera clients: {', '.join(camera_names)}")
    else:
        env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    validate_leader_follower_dims(env, agent)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    from gello.utils.control_utils import SaveInterface, run_control_loop

    # Initialize save interface if requested
    save_interface = None
    if args.use_save_interface:
        save_root = Path(args.left_config_path).parents[1] / "data"
        print(f"Save root: {save_root}")
        save_interface = SaveInterface(
            data_dir=save_root,
            agent_name=agent.__class__.__name__,
            expand_user=True,
            on_start_recording=robot_client.reference_gripper,
        )

    # Run main control loop
    run_control_loop(env, agent, save_interface)


if __name__ == "__main__":
    main()
