import time
from typing import Optional

import kinpy as kp
import numpy as np
from loguru import logger
from pymycobot.mycobot import MyCobot

from .settings import MyCobotSettings


class MyCobotController:
    def __init__(self, settings: MyCobotSettings):
        self._mycobot = MyCobot(settings.port, settings.baud)
        self._suction_pin = settings.suction_pin
        self._default_speed = settings.default_speed
        self._default_z_speed = settings.default_z_speed
        self._command_timeout = settings.command_timeout
        self._use_gravity_compensation = settings.use_gravity_compensation
        self._sim = kp.build_serial_chain_from_urdf(open(settings.urdf_path).read(), settings.end_effector_name)
        self._current_position = self._mycobot.get_angles()
        self.positions = {place.name: place.position for place in settings.places}
        if "home" not in self.positions:
            # set default home position
            self.positions["home"] = [0, 20, -130, 20, 0, 0]
        if "capture" not in self.positions:
            # set default capture position
            self.positions["capture"] = [0, 0, -30, -60, 0, -45]
        self.capture_coord = self._calc_camera_lens_coords_on_capture_position(settings.urdf_path)
        self.end_effector_height = settings.end_effector_height  # pump head offset
        self.object_height = settings.object_height
        self.release_height = settings.release_height
        self._detections: list[tuple[float, float]] = []

    def _calc_camera_lens_coords_on_capture_position(self, urdf_path: str) -> kp.Transform:
        sim_for_lens = kp.build_serial_chain_from_urdf(open(urdf_path).read(), "camera_lens")
        return sim_for_lens.forward_kinematics(np.deg2rad(self.positions["capture"]))

    def calc_gravity_compensation(self, angles: list) -> np.ndarray:
        if not self._use_gravity_compensation:
            return np.zeros(6)
        k = np.array([0.0, 0.0, -0.15, -0.35, 0.0, 0.0])
        mat = self._sim.jacobian(np.deg2rad(angles))
        d_ang = np.rad2deg(np.dot(mat.T, np.array([0, 0, -9.8, 0, 0, 0]))) * k
        return d_ang

    def set_detections(self, detections: list[tuple[float, float]]) -> None:
        self._detections = detections

    def clear_detections(self) -> None:
        self._detections = []

    def current_coords(self) -> kp.Transform:
        return self._sim.forward_kinematics(np.deg2rad(self._current_position))

    def move_to_xy(self, x: float, y: float, speed: Optional[float] = None) -> list[str]:
        """Move to absolute position xy"""
        coords = self.current_coords()
        coords.pos[0] = x
        coords.pos[1] = y
        self.move_to_coords(coords, speed)
        return [f"move_to_xy({x}, {y}, {speed})"]

    def move_to_z(self, z: float, speed: Optional[float] = None) -> list[str]:
        """Move to absolute position z"""
        coords = self.current_coords()
        coords.pos[2] = z
        self.move_to_coords(coords, speed or self._default_z_speed)
        return [f"move_to_z({z}, {speed})"]

    def move_to_coords(self, coords: kp.Transform, speed: Optional[float] = None) -> list[str]:
        position = self._sim.inverse_kinematics(coords, np.deg2rad(self._current_position))
        self._current_position = np.rad2deg(position)
        self._mycobot.sync_send_angles(
            (np.array(self._current_position) + self.calc_gravity_compensation(self._current_position)).tolist(),
            speed or self._default_speed,
            self._command_timeout,
        )
        return [f"move_to_coords({coords}, {speed})"]

    def move_to_object(self, object_no: int, speed: Optional[float] = None) -> list[str]:
        logger.info("[MyCobotController] Move to Object No. {}".format(object_no))
        detection = (
            np.array([self._detections[object_no][0], self._detections[object_no][1]]) + self.capture_coord.pos[:2]
        )
        logger.info("[MyCobotController] Object pos: {} {}".format(detection[0], detection[1]))
        self.move_to_xy(detection[0], detection[1], speed)
        return [f"move_to_object({object_no}, {speed})"]

    def move_to_place(self, place_name: str, speed: Optional[float] = None) -> list[str]:
        logger.info("[MyCobotController] Move to Place {}".format(place_name))
        self._current_position = self.positions[place_name]
        self._mycobot.sync_send_angles(
            (np.array(self._current_position) + self.calc_gravity_compensation(self._current_position)).tolist(),
            speed or self._default_speed,
            self._command_timeout,
        )
        return [f"move_to_place({place_name}, {speed})"]

    def grab(self, speed: Optional[float] = None) -> list[str]:
        logger.info("[MyCobotController] Grab to Object")
        current_pos = self.current_coords().pos
        messages = []
        messages.extend(self.move_to_z(self.object_height + self.end_effector_height, speed))
        self._mycobot.set_basic_output(self._suction_pin, 0)
        messages.append("set_suction(0)")
        time.sleep(2)
        messages.extend(self.move_to_z(current_pos[2], speed))
        return messages

    def release(self, speed: Optional[float] = None) -> list[str]:
        logger.info("[MyCobotController] Release")
        current_pos = self.current_coords().pos
        messages = []
        messages.extend(self.move_to_z(self.release_height + self.end_effector_height, speed))
        self._mycobot.set_basic_output(self._suction_pin, 1)
        messages.append("set_suction(1)")
        time.sleep(1)
        messages.extend(self.move_to_z(current_pos[2], speed))
        return messages
