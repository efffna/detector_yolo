from pathlib import Path
from pydantic import BaseSettings, Field


class ObjectDetectorConfig(BaseSettings):
    # model config
    model_path: Path = Field(default="models/yolov5n_v3.onnx", env="OBJECT_DETECTOR_MODEL_PATH")
    img_width: int = Field(default=640, env="DETECTOR_IMG_WIDTH")
    img_height: int = Field(default=640, env="DETECTOR_IMG_HEIGHT")
    conf_threshold: float = 0.7
    iou_threshold: float = 0.45


class AppConfig(BaseSettings):
    object_detector: ObjectDetectorConfig = ObjectDetectorConfig()