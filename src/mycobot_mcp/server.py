import argparse
import base64
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any

import cv2
import numpy as np
from loguru import logger

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent, ImageContent

from .robot_operator import CaptureResult, OperationCode, RobotOperator
from .settings import MyCobotMCPSettings


_settings: MyCobotMCPSettings | None = None


def _ndarray_to_base64(image: np.ndarray) -> tuple[str, str]:
    """Convert a numpy array to a base64 string"""
    image_base64_png = base64.b64encode(cv2.imencode(".png", image)[1]).decode("utf-8")
    return image_base64_png, "image/png"


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    try:
        logger.info("myCobotMCP server starting up")
        try:
            _ = get_robot_operator()
            logger.info("Successfully connected to myCobot on startup")
        except Exception as e:
            logger.warning(f"Could not connect to myCobot on startup: {str(e)}")
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _robot_operator
        if _robot_operator:
            logger.info("Disconnecting from myCobot on shutdown")
            _robot_operator = None
        logger.info("myCobotMCP server shut down")


mcp = FastMCP(
    "MyCobotMCP",
    description="MyCobot integration through the Model Context Protocol",
    lifespan=server_lifespan,
)

_robot_operator: RobotOperator | None = None
_detection_result_history: deque[CaptureResult] = deque(maxlen=10)


def get_robot_operator():
    """Get or create a persistent FreeCAD connection"""
    global _robot_operator, _settings
    if _robot_operator is None:
        _robot_operator = RobotOperator(settings=_settings)
    return _robot_operator


def get_detection_result_history():
    """Get the detection result history"""
    global _detection_result_history
    return _detection_result_history


@mcp.tool()
def get_robot_settings(ctx: Context) -> list[TextContent]:
    """Get the robot settings"""
    global _settings
    settings = _settings or MyCobotMCPSettings()
    return [TextContent(type="text", text=settings.model_dump_json())]


@mcp.tool()
def capture(ctx: Context) -> ImageContent:
    """Capture a camera image"""
    robot_operator = get_robot_operator()
    image = robot_operator.capture_image()
    image_base64, mime_type = _ndarray_to_base64(image)
    return ImageContent(type="image", data=image_base64, mimeType=mime_type)


@mcp.tool()
def capture_and_detect(ctx: Context, object_lists: list[str]) -> list[TextContent | ImageContent]:
    """Capture a camera image and detect objects in the image

    Args:
        object_lists (list[str]): The list of object names to detect in English

    Returns:
        list[TextContent | ImageContent]: The detection results
    """
    robot_operator = get_robot_operator()
    result = robot_operator.capture_and_detect(object_lists)
    if isinstance(result, list):
        return result
    get_detection_result_history().append(result)
    image_base64, mime_type = _ndarray_to_base64(result.annotated_image)
    detected_coords = robot_operator.detection_to_coords(result.detections)
    return [
        TextContent(
            type="text",
            text=str(
                [
                    {
                        "object_no": i,
                        "category": detection.category,
                        "score": detection.score,
                        "coords": detected_coords[i],
                    }
                    for i, detection in enumerate(result.detections)
                ]
            ),
        ),
        ImageContent(type="image", data=image_base64, mimeType=mime_type),
    ]


@mcp.tool()
def capture_with_grid(ctx: Context) -> ImageContent:
    """Capture a camera image with a grid"""
    robot_operator = get_robot_operator()
    image = robot_operator.capture_with_grid()
    image_base64, mime_type = _ndarray_to_base64(image)
    return ImageContent(type="image", data=image_base64, mimeType=mime_type)


@mcp.tool()
def run(ctx: Context, code: list[OperationCode]) -> list[TextContent | ImageContent]:
    """Run the robot operator with the given operation codes

    Args:
        code (list[OperationCode]): The list of operation codes to run

    Example:
        You want the robot to grab a No.0 object, then move it to the drop place defined in the robot settings.
        ```json
        [
            {"action": "move_to_object", "object_no": 0},
            {"action": "grab"},
            {"action": "move_to_place", "place_name": "drop"},
            {"action": "release"},
        ]
        ```

        You want the robot to grab a No.0 object, then drop it to the No.1 object(storage or other place).
        ```json
        [
            {"action": "move_to_object", "object_no": 0},
            {"action": "grab"},
            {"action": "move_to_object", "object_no": 1},
            {"action": "release"},
        ]
        ```

    Returns:
        list[TextContent | ImageContent]: The result of the operation
    """
    robot_operator = get_robot_operator()
    if len(get_detection_result_history()) == 0:
        return [TextContent(type="text", text="No detection results available")]
    detections = get_detection_result_history()[-1].detections
    result = robot_operator.run(code, detections)
    return [TextContent(type="text", text=str(result))]


@mcp.prompt()
def code_generation_strategy() -> str:
    return """You are a skilled robot arm operator.
The robot arm is equipped with a camera.
The camera is mounted at the tip of the arm and is facing the floor.
Following given instructions, you can operate the robot arm to pick up, release, or move objects.

1. First, retrieve the robot's settings. (`get_robot_settings`)
2. Next, capture an image from the camera. (`capture`)
3. Analyze the camera image to determine which object to detect. (`capture_and_detect`, `capture_with_grid`)
   * `capture_with_grid` captures an image from the camera and overlays a grid on the image.
   * `capture_and_detect` captures an image from the camera and overlays detected objects on the image.
4. Based on the information of the detected objects, generate operation code to allow the robot arm to pick up, release, or move the objects. (`run`)
   * If the target position for the robot arm is an object, use `move_to_object` and specify the object number.
   * If the target position is a predefined place, use `move_to_place` and specify the place name.
   * If the target position is a coordinate on the camera grid, use `move_to_xy_on_capture_coord` and specify the coordinate on the camera grid.
   In this case, the current position of the robot arm corresponds to the center of the camera image.
"""


def main():
    """Run the MCP server"""
    global _settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-path", type=str, default=None)
    args = parser.parse_args()
    if args.settings_path:
        _settings = MyCobotMCPSettings.model_validate_json(open(args.settings_path).read())
    mcp.run()
