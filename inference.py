import os
import argparse
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from tqdm import trange

from service import MonitoringServiceInterface


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser('Track pigs on video')
    parser.add_argument('--video_path', '-v', type=str, help='mp4 test file path')
    parser.add_argument('--out_path', '-op', type=str, help='output file path')
    return parser


def build_monitoring_service():
    from config import AppConfig
    from infrastructure.detection.yolov5 import YoloV5Detector
    from service import DummyBoxMonitoringService
    
    app_config = AppConfig()
    detector_config = app_config.object_detector
    object_detector = YoloV5Detector(detector_config)
    object_monitoring_service = DummyBoxMonitoringService(object_detector)
    return object_monitoring_service


def draw_predictions(frame, coords):
    fontScale = 1
    thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    for coord in coords:
        x_min, y_min, x_max, y_max = coord.x_min, coord.y_min, coord.x_max, coord.y_max
        color = (255, 0, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        cv2.putText(frame, f'{coord.activity}', (x_min, y_max), font, fontScale, color, thickness, cv2.LINE_AA)
        count += 1
    frame = (frame * 0.75).astype(np.uint8)
    return frame


def process_video(video_path: Path, monitoring_service: MonitoringServiceInterface):
    frames_preds = []
    capture = cv2.VideoCapture(video_path)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(f'output/{os.path.basename(video_path)}', fourcc, fps, (width, height))

    for _ in trange(length, desc='Processing video...'):
        ret, frame = capture.read()
        if frame is None:
            break
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracked_boxes = monitoring_service.process_frame(new_frame)
        frames_preds.append(tracked_boxes)
        new_frame = draw_predictions(new_frame, tracked_boxes)
        writer.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))

    writer.release()     
    capture.release()
    return frames_preds


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    video_path = args.video_path
    monitoring_service = build_monitoring_service()
    frames_preds = process_video(video_path, monitoring_service)