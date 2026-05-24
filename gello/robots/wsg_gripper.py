import sys
import time
from pathlib import Path
from typing import Optional


def _add_wsg_driver_path() -> None:
    repo_root = None
    for parent in Path(__file__).resolve().parents:
        if (parent / "backup" / "src" / "weiss_gripper_ieg76").exists():
            repo_root = parent
            break
    if repo_root is None:
        raise ImportError("Could not find backup/src/weiss_gripper_ieg76")

    driver_path = (
        repo_root
        / "backup"
        / "src"
        / "weiss_gripper_ieg76"
        / "weiss_gripper_ieg76"
        / "weiss_gripper_ieg76"
    )
    if str(driver_path) not in sys.path:
        sys.path.append(str(driver_path))


class WSGGripper:
    """Small adapter for the Weiss WSG gripper teleop driver."""

    def __init__(
        self,
        port_name: str = "/dev/ttyACM0",
        max_width_mm: float = 110.0,
    ) -> None:
        _add_wsg_driver_path()
        from driver_teleop import Driver

        self._driver = Driver(serial_port_name=port_name)
        self._driver.serial_port_comm.add_flags_observer(self)
        self._max_width_mm = max_width_mm
        self._live_position: Optional[float] = None
        self._last_update_time = 0.0
        self._last_state: Optional[bool] = None
        print(f"[WSG] Connected on {port_name}")

    def update_flags(self, flags_dict) -> None:
        self._live_position = flags_dict.get("POS", None)
        self._last_update_time = time.time()

    def get_current_position(self, max_age_s: float = 2.0) -> float:
        if self._live_position is None:
            if self._last_state is not None:
                return 1.0 if self._last_state else 0.0
            return 0.0
        if time.time() - self._last_update_time > max_age_s:
            if self._last_state is not None:
                return 1.0 if self._last_state else 0.0
        position = float(self._live_position) / self._max_width_mm
        return min(max(position, 0.0), 1.0)

    def move(self, state_value: float, speed: int = 255, force: int = 10) -> None:
        del speed, force
        open_gripper = state_value > 0.5
        if self._last_state == open_gripper:
            return
        if open_gripper:
            print("[WSG] Command: OPEN")
            self._driver.reference()
        else:
            print("[WSG] Command: CLOSE")
            self._driver.close(position=0.0)
        self._last_state = open_gripper

    def reference(self) -> None:
        print("[WSG] Requesting reference")
        self._driver.reference()
        self._last_state = True

    def close_connection(self) -> None:
        print("[WSG] Shutting down")
        self._driver.shutdown()
