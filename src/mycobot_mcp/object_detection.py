import os

import cv2
import numpy as np
import supervision as sv
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from pydantic import BaseModel

from .rle_utils import rle_to_array


class Mask(BaseModel):
    counts: str
    size: list[int]


class DetectionResult(BaseModel):
    bbox: list[float]
    category: str
    mask: Mask
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2


class ObjectDetection:
    def __init__(self):
        token = os.getenv("DDS_API_TOKEN")
        if not token:
            raise ValueError("DDS_API_TOKEN is not set")
        config = Config(token)
        self.client = Client(config)

    def detect(self, image_path: str, object_lists: list[str]) -> list[DetectionResult]:
        image_url = self.client.upload_file(image_path)

        v2_task = V2Task(
            api_path="/v2/task/dinox/detection",
            api_body={
                "model": "DINO-X-1.0",
                "image": image_url,
                "prompt": {
                    "type": "text",
                    "text": " . ".join(object_lists)
                },
                "targets": ["bbox", "mask"],
                "bbox_threshold": 0.25,
                "iou_threshold": 0.8
            }
        )

        self.client.run_task(v2_task)
        result = v2_task.result
        objects = result["objects"]
        return [DetectionResult(**obj) for obj in objects]

    def visualize(self, image_path: str, objects: list[DetectionResult], with_mask: bool = False) -> np.ndarray:
        if not objects:
            return cv2.imread(image_path)

        classes = [obj.category.strip().lower() for obj in objects]
        class_name_to_id = {name: id for id, name in enumerate(classes)}

        boxes = []
        masks = []
        confidences = []
        class_names = []
        class_ids = []

        for obj in objects:
            boxes.append(obj.bbox)
            masks.append(
                rle_to_array(
                    obj.mask.counts,
                    obj.mask.size[0] * obj.mask.size[1]
                ).reshape(obj.mask.size)
            )
            confidences.append(obj.score)
            cls_name = obj.category.lower().strip()
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])

        boxes = np.array(boxes)
        masks = np.array(masks)
        class_ids = np.array(class_ids)
        labels = [
            f"{id} {class_name}"
            for id, class_name
            in enumerate(class_names)
        ]

        img = cv2.imread(image_path)
        detections = sv.Detections(
            xyxy = boxes,
            mask = masks.astype(bool),
            class_id = class_ids,
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        if with_mask:
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
        else:
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
        return annotated_frame

