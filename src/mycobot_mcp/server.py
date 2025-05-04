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
    return [
        TextContent(type="text", text=str(robot_operator.detection_to_coords(result.detections))),
        ImageContent(type="image", data=image_base64, mimeType=mime_type),
    ]


@mcp.tool()
def run(ctx: Context, code: list[OperationCode]) -> list[TextContent | ImageContent]:
    """Run the robot operator

    Args:
        code (list[OperationCode]): The list of operation codes to run

    Example:
        You want the robot to grab a No.0 object, then move it to the drop place.
        ```json
        [
            {"action": "move_to_object", "object_no": 0},
            {"action": "grab"},
            {"action": "move_to_place", "place_name": "drop"},
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
    return """You are a smart robot operator.
You are given a list of object detections.
You need to generate a list of operation codes to move the robot to grab the object and move it to the drop place.

1. First, get the robot settings. (get_robot_settings)
2. Next, get the camera image. (capture)
3. Analyze the captured image and create a list of objects to detect, and perform object detection. (capture_and_detect)
4. Based on the information of the detected objects, generate a list of operation codes to move the robot to grab the object and move it to the drop place.
5. Execute the generated operation codes. (run)
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
