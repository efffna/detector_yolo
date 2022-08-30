from typing import Protocol, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from infrastructure.detection.model import DetectionMeta, DetectorInterface


class MonitoringServiceInterface(ABC):
    def __init__(self, object_detector: DetectorInterface):
        self.object_detector = object_detector

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> List[DetectionMeta]:
        raise NotImplementedError


class DummyBoxMonitoringService(MonitoringServiceInterface):
    def __init__(self, object_detector: DetectorInterface):
        super().__init__(object_detector)

    def process_frame(self, frame: np.ndarray) -> List[DetectionMeta]:
        detections_meta = self.object_detector.predict(frame=frame)
        return detections_meta
