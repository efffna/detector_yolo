from typing import List
from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionMeta:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float
    class_name: str = "Box"
    activity: int = 0

    def __post_init__(self):
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.bbox = (self.x_min, self.y_min, self.width, self.height)


class DetectorInterface(ABC):
  @abstractmethod
  def predict(self, frame: np.ndarray) -> List[DetectionMeta]:
    raise NotImplementedError

  @abstractmethod
  def _load_model(self):
    raise NotImplementedError

    
  def _load_model(self):
    pass
