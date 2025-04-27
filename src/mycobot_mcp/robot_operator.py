import tempfile
from typing import Literal

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from .camera import FlatPixelToWorld
from .object_detection import DetectionResult, ObjectDetection
from .robot_controller import MyCobotController
from .settings import MyCobotMCPSettings


class CaptureResult(BaseModel):
    image: np.ndarray
    annotated_image: np.ndarray
    detections: list[DetectionResult]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OperationCode(BaseModel):
    action: Literal["grab", "release", "move_to_object", "move_to_place"] = Field(
        description="The action to perform"
    )
    object_no: int | None = Field(
        default=None, description="If action is move_to_object, the number of the object to move to. 0-indexed."
    )
    place_name: str | None = Field(
        default=None, description="If action is move_to_place, the name of the place to move to"
    )


def swap_xy(xy: tuple[float, float]) -> tuple[float, float]:
    return xy[1], xy[0]


class RobotOperator:
    def __init__(self, settings: MyCobotMCPSettings | None = None):
        settings = settings or MyCobotMCPSettings()
        self._robot_controller = MyCobotController(settings.mycobot_settings)
        self._cap = cv2.VideoCapture(settings.camera_id)
        self._camera_parameter = FlatPixelToWorld.from_camera_parameters_path(settings.camera_parameter_path)
        test_image = self.capture_image()
        self._cam_center = (test_image.shape[1] / 2, test_image.shape[0] / 2)
        self._object_detection = ObjectDetection()

    def capture_image(self) -> np.ndarray | None:
        self._robot_controller.move_to_place("capture")
        ret, frame = self._cap.read()
        if not ret:
            logger.error("Failed to capture image")
            return None
        frame = cv2.undistort(
            frame,
            self._camera_parameter.matrix,
            self._camera_parameter.distortion,
            None,
        )
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def capture_and_detect(self, object_lists: list[str]) -> CaptureResult | list[str]:
        try:
            image = self.capture_image()
            if image is None:
                return ["Error capturing image"]
            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                cv2.imwrite(f.name, image)
                detections = self._object_detection.detect(f.name, object_lists)
                annotated_image = self._object_detection.visualize(f.name, detections)
                return CaptureResult(
                    image=image, annotated_image=annotated_image, detections=detections
                )
        except Exception as e:
            logger.error(f"Error capturing and detecting: {e}")
            return ["Error capturing and detecting:"] + [str(e)]

    def detection_to_coords(self, detections: list[DetectionResult]) -> list[tuple[float, float]]:
        height = self._robot_controller.capture_coord.pos[2]
        return [
            swap_xy(self._camera_parameter.uv_to_xy(detection.center[0], detection.center[1], height))
            for detection in detections
        ]

    def run(self, code: list[OperationCode], detections: list[DetectionResult]) -> list[str]:
        detections = self.detection_to_coords(detections)
        self._robot_controller.set_detections(detections)
        # First, move to home position
        self._robot_controller.move_to_place("home")
        try:
            messages = []
            for operation in code:
                if operation.action == "grab":
                    messages.extend(self._robot_controller.grab())
                elif operation.action == "release":
                    messages.extend(self._robot_controller.release())
                elif operation.action == "move_to_object":
                    messages.extend(self._robot_controller.move_to_object(operation.object_no))
                elif operation.action == "move_to_place":
                    messages.extend(self._robot_controller.move_to_place(operation.place_name))
            self._robot_controller.clear_detections()
            return ["Success running code:"] + messages
        except Exception as e:
            logger.error(f"Error running code: {e}")
            return ["Error running code:"] + [str(e)]
