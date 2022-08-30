from typing import List
from pathlib import Path
import onnxruntime
import numpy as np

from config import ObjectDetectorConfig
from infrastructure.detection.model import DetectionMeta, DetectorInterface
from infrastructure.detection.utils import nms, scale_coords, letterbox


class YoloV5Detector(DetectorInterface):
    def __init__(self, config: ObjectDetectorConfig):
      self.model_path = config.model_path
      self.width = config.img_width
      self.height = config.img_height
      self.score_threshold = config.conf_threshold
      self.iou_threshold = config.iou_threshold
      self._load_model(self.model_path)

    def _load_model(self, model_path: Path):
        self.session = onnxruntime.InferenceSession(str(model_path))
        print(f"Detector inference on device: {onnxruntime.get_device()}")

    def _preprocess_input(self, img: np.ndarray):
        img, ratio, pad = letterbox(img,new_shape=(self.height, self.width), auto=False)
        tensor = img[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        tensor = tensor.astype(np.float32)
        tensor /= 255.0
        if len(tensor.shape) == 3:
               tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def predict(self, frame: np.ndarray) -> List[DetectionMeta]:
        resized_image = self._preprocess_input(frame)
        pred = self.session.run([self.session.get_outputs()[0].name], 
                                {self.session.get_inputs()[0].name: resized_image})[0]
        pred = nms(pred, self.score_threshold, self.iou_threshold)
        output = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(resized_image.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, score, cls_id in reversed(det):
                x_min, y_min, x_max, y_max = list(map(lambda c: int(c.item()), xyxy))
                coords = DetectionMeta(x_min, y_min, x_max, y_max, score)
                output.append(coords)
        return output
