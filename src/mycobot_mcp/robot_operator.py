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
    action: Literal["grab", "release", "move_to_object", "move_to_place", "move_to_xy_on_capture_coord"] = Field(
        description="The action to perform"
    )
    object_no: int | None = Field(
        default=None, description="If action is move_to_object, the number of the object to move to. 0-indexed."
    )
    place_name: str | None = Field(
        default=None, description="If action is move_to_place, the name of the place to move to"
    )
    xy: tuple[float, float] | None = Field(
        default=None, description="If action is move_to_xy_on_capture_coord, the xy coordinates to move to"
    )

def swap_xy(xy: tuple[float, float]) -> tuple[float, float]:
    return xy[1], xy[0]


def draw_real_world_grid(
    image: np.ndarray,
    camera_parameter: FlatPixelToWorld,
    height: float,
    grid_size: float = 0.01,  # [m]
    grid_color: tuple[int, int, int] = (0, 0, 255),
    grid_thickness: int = 1,
    draw_axes: bool = True,
) -> np.ndarray:
    copied_image = image.copy()
    height, width = copied_image.shape[:2]

    corners_uv = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    xs, ys = zip(*[camera_parameter.uv_to_xy(u, v, height) for (u, v) in corners_uv])
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    x_pos = np.arange(0, max_x + 1e-9, grid_size)
    x_neg = np.arange(-grid_size, min_x - 1e-9, -grid_size)
    x_lines = np.concatenate((x_neg, x_pos))

    y_pos = np.arange(0, max_y + 1e-9, grid_size)
    y_neg = np.arange(-grid_size, min_y - 1e-9, -grid_size)
    y_lines = np.concatenate((y_neg, y_pos))

    for x in x_lines:
        cv2.line(copied_image, camera_parameter.xy_to_uv(x, min_y), camera_parameter.xy_to_uv(x, max_y), grid_color, grid_thickness)
    for y in y_lines:
        cv2.line(copied_image, camera_parameter.xy_to_uv(min_x, y), camera_parameter.xy_to_uv(max_x, y), grid_color, grid_thickness)

    if draw_axes:
        cx, cy = width / 2, height / 2
        origin_px = (int(round(cx)), int(round(cy)))

        axis_len = 5 * grid_size
        x_end_px = camera_parameter.xy_to_uv(axis_len, 0)    # +X
        y_end_px = camera_parameter.xy_to_uv(0, axis_len)    # +Y

        cv2.arrowedLine(copied_image, origin_px, x_end_px, (0, 0, 255), grid_thickness + 1, tipLength=0.05)
        cv2.putText(copied_image, "X+", (x_end_px[0] + 5, x_end_px[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.arrowedLine(copied_image, origin_px, y_end_px, (255, 0, 0), grid_thickness + 1, tipLength=0.05)
        cv2.putText(copied_image, "Y+", (y_end_px[0] + 5, y_end_px[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.drawMarker(copied_image, origin_px, (0, 255, 255), cv2.MARKER_CROSS, 15, grid_thickness + 1)

    return copied_image


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

    def capture_with_grid(self) -> np.ndarray:
        image = self.capture_image()
        if image is None:
            return None
        return draw_real_world_grid(
            image, self._camera_parameter, self._robot_controller.capture_coord.pos[2]
        )

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
                elif operation.action == "move_to_xy_on_capture_coord":
                    messages.extend(self._robot_controller.move_to_xy_on_capture_coord(operation.xy[0], operation.xy[1]))
            self._robot_controller.clear_detections()
            return ["Success running code:"] + messages
        except Exception as e:
            logger.error(f"Error running code: {e}")
            return ["Error running code:"] + [str(e)]
