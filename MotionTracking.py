#!/usr/bin/env python3
"""
Motion Tracking (Robust with Deep Learning)
---------------------------------------------
Description:
  This script performs video motion tracking to detect and track moving objects.
  It supports several modes:
    - "mask": outputs the background subtraction mask video.
    - "bounding": outputs a video with bounding boxes drawn on detected motion.
    - "path": outputs a video with a tracking path drawn using temporal smoothing.
    - "dnn": uses a deep learning detector (YOLO) to detect persons and draw bounding boxes.
GPU acceleration is used where available.

Author: Sabneet Bains
License: MIT License
"""

import argparse
import logging
import os
from typing import Optional, List

import cv2 as cv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionTracking:
    """Motion tracking class for robust detection and tracking."""

    def __init__(self, video_path: str, min_area: float = 500.0, smoothing: float = 0.3) -> None:
        """
        Initialize the MotionTracking object.

        Parameters:
            video_path (str): Path to the input video file or image sequence pattern.
            min_area (float): Minimum contour area to consider as valid motion.
            smoothing (float): Smoothing factor (0.0-1.0) for temporal filtering of coordinates.
        """
        self.video = cv.VideoCapture(video_path)
        if not self.video.isOpened():
            logger.error("Failed to open video: %s", video_path)
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.min_area = min_area
        self.smoothing = smoothing  # e.g. new detection weight 0.3, previous 0.7

        # Check for GPU acceleration and set up background subtractor accordingly.
        self.gpu_available: bool = cv.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            try:
                self.background = cv.cuda.createBackgroundSubtractorMOG2()
                logger.info("Using GPU background subtractor.")
            except Exception as e:
                logger.warning("GPU background subtractor unavailable: %s. Falling back to CPU.", e)
                self.background = cv.createBackgroundSubtractorMOG2()
        else:
            self.background = cv.createBackgroundSubtractorMOG2()

        self.frame_width: int = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height: int = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = self.video.get(cv.CAP_PROP_FPS)
        self.frame_count: int = int(self.video.get(cv.CAP_PROP_FRAME_COUNT))

        # Placeholders for processing
        self.kernel: Optional[np.ndarray] = None
        self.frame: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None

        self.contours: Optional[List[np.ndarray]] = None
        self.areas: Optional[List[float]] = None
        self.max_contour: Optional[np.ndarray] = None

        # Bounding box parameters and smoothing state
        self.x: Optional[int] = None
        self.y: Optional[int] = None
        self.w: Optional[int] = None
        self.h: Optional[int] = None
        self.prev_x: Optional[int] = None
        self.prev_y: Optional[int] = None

    def preprocess(self, morph_size: int, blur_size: int) -> bool:
        """
        Read a frame, apply background subtraction, morphological opening, and median blur.

        Parameters:
            morph_size (int): Size of the structuring element for morphological opening.
            blur_size (int): Kernel size for median blur.

        Returns:
            bool: True if a frame was successfully read and processed; otherwise, False.
        """
        ret, frame = self.video.read()
        if not ret:
            return False
        self.frame = frame

        # Background subtraction using GPU if available.
        if self.gpu_available:
            try:
                gpu_frame = cv.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_mask = self.background.apply(gpu_frame)
                mask = gpu_mask.download()
            except Exception as e:
                logger.warning("GPU background subtraction failed: %s. Using CPU.", e)
                mask = self.background.apply(frame)
        else:
            mask = self.background.apply(frame)
        self.mask = mask

        # Create the structuring element.
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_size, morph_size))
        # Apply morphological opening.
        if self.gpu_available:
            try:
                gpu_mask = cv.cuda_GpuMat()
                gpu_mask.upload(self.mask)
                morph_filter = cv.cuda.createMorphologyFilter(cv.MORPH_OPEN, self.mask.dtype, self.kernel)
                gpu_mask = morph_filter.apply(gpu_mask)
                mask = gpu_mask.download()
            except Exception as e:
                logger.warning("GPU morphological open failed: %s. Using CPU.", e)
                mask = cv.morphologyEx(self.mask, cv.MORPH_OPEN, self.kernel)
        else:
            mask = cv.morphologyEx(self.mask, cv.MORPH_OPEN, self.kernel)
        # Apply median blur.
        if self.gpu_available:
            try:
                gpu_mask = cv.cuda_GpuMat()
                gpu_mask.upload(mask)
                gpu_mask = cv.cuda.medianBlur(gpu_mask, blur_size)
                mask = gpu_mask.download()
            except Exception as e:
                logger.warning("GPU median blur failed: %s. Using CPU.", e)
                mask = cv.medianBlur(mask, blur_size)
        else:
            mask = cv.medianBlur(mask, blur_size)
        self.mask = mask
        return True

    def find_contours(self) -> None:
        """Find contours in the current mask and compute their areas."""
        cnts, _ = cv.findContours(self.mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.contours = cnts
        self.areas = [cv.contourArea(c) for c in cnts] if cnts is not None else []

    def check_contours(self) -> bool:
        """
        Select the largest contour and compute its bounding rectangle if it exceeds min_area.
        Also applies temporal smoothing to the bounding box coordinates.

        Returns:
            bool: True if a valid contour is detected, False otherwise.
        """
        if not self.areas or len(self.areas) == 0:
            return False
        max_idx = int(np.argmax(self.areas))
        if self.areas[max_idx] < self.min_area:
            return False
        self.max_contour = self.contours[max_idx]
        x_new, y_new, w, h = cv.boundingRect(self.max_contour)
        if self.prev_x is None or self.prev_y is None:
            self.x, self.y = x_new, y_new
        else:
            self.x = int((1 - self.smoothing) * self.prev_x + self.smoothing * x_new)
            self.y = int((1 - self.smoothing) * self.prev_y + self.smoothing * y_new)
        self.w, self.h = w, h
        self.prev_x, self.prev_y = self.x, self.y
        return True

    def create_mask(self, morph_size: int = 5, blur_size: int = 5, output_file: str = 'masked.mp4') -> None:
        """
        Create a video showing the background subtraction mask.

        Parameters:
            morph_size (int): Morphological kernel size.
            blur_size (int): Median blur kernel size.
            output_file (str): Output video filename.
        """
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_file, fourcc, self.fps, (self.frame_width, self.frame_height), isColor=False)
        while self.preprocess(morph_size, blur_size):
            writer.write(self.mask)
        writer.release()
        logger.info("Masked video saved to %s", output_file)

    def create_bounding_box(self, morph_size: int = 3, blur_size: int = 5, output_file: str = 'bounded.mp4') -> None:
        """
        Create a video with bounding boxes drawn around valid moving objects.

        Parameters:
            morph_size (int): Morphological kernel size.
            blur_size (int): Median blur kernel size.
            output_file (str): Output video filename.
        """
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_file, fourcc, self.fps, (self.frame_width, self.frame_height))
        while self.preprocess(morph_size, blur_size):
            self.find_contours()
            if self.check_contours():
                cv.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
            writer.write(self.frame)
        writer.release()
        logger.info("Bounding box video saved to %s", output_file)

    def create_path(self, starting_frame: int, ending_frame: int, morph_size: int = 5, blur_size: int = 11, output_file: str = 'tracked.mp4') -> None:
        """
        Create a video showing the tracking path of the largest moving object,
        applying temporal smoothing to the detected coordinates.

        Parameters:
            starting_frame (int): The frame index to start tracking.
            ending_frame (int): The frame index to end tracking.
            morph_size (int): Morphological kernel size.
            blur_size (int): Median blur kernel size.
            output_file (str): Output video filename.
        """
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_file, fourcc, self.fps, (self.frame_width, self.frame_height))
        frame_idx = 0
        while frame_idx < starting_frame:
            ret, _ = self.video.read()
            if not ret:
                break
            frame_idx += 1
        while frame_idx < ending_frame and self.preprocess(morph_size, blur_size):
            self.find_contours()
            if self.check_contours():
                cv.line(self.frame, (self.prev_x, self.prev_y), (self.x, self.y), (0, 0, 255), 2)
            writer.write(self.frame)
            frame_idx += 1
        writer.release()
        cv.imwrite('tracked.jpg', self.frame)
        logger.info("Tracked video saved to %s and snapshot saved to tracked.jpg", output_file)

    def create_dnn_path(self, starting_frame: int, ending_frame: int, yolo_cfg: str, yolo_weights: str,
                        yolo_names: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4,
                        output_file: str = 'dnn_tracked.mp4') -> None:
        """
        Create a video using a deep learning detector (YOLO) to detect persons and draw bounding boxes.
        Only detections with confidence above conf_threshold are considered, and non-max suppression is applied.

        Parameters:
            starting_frame (int): Frame index to start processing.
            ending_frame (int): Frame index to end processing.
            yolo_cfg (str): Path to YOLO config file.
            yolo_weights (str): Path to YOLO weights file.
            yolo_names (str): Path to file with class names.
            conf_threshold (float): Confidence threshold.
            nms_threshold (float): Non-max suppression threshold.
            output_file (str): Output video filename.
        """
        # Load class names.
        with open(yolo_names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        # Load YOLO network.
        net = cv.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
        # Use GPU if available.
        if self.gpu_available:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_file, fourcc, self.fps, (self.frame_width, self.frame_height))
        frame_idx = 0
        # Skip frames until starting_frame.
        while frame_idx < starting_frame:
            ret, _ = self.video.read()
            if not ret:
                break
            frame_idx += 1

        # Process frames using DNN detection.
        while frame_idx < ending_frame:
            ret, frame = self.video.read()
            if not ret:
                break
            blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes = []
            confidences = []
            class_ids = []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # Filter for person class (assumed "person" is in classes) and threshold.
                    if classes[class_id].lower() == "person" and confidence > conf_threshold:
                        center_x = int(detection[0] * self.frame_width)
                        center_y = int(detection[1] * self.frame_height)
                        w = int(detection[2] * self.frame_width)
                        h = int(detection[3] * self.frame_height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(frame, f"{classes[class_ids[i]]} {confidences[i]:.2f}", (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            writer.write(frame)
            frame_idx += 1
        writer.release()
        cv.imwrite('dnn_tracked.jpg', frame)
        logger.info("DNN-tracked video saved to %s and snapshot to dnn_tracked.jpg", output_file)

def main() -> None:
    parser = argparse.ArgumentParser(description="Robust Motion Tracking with Optional Deep Learning Detection.")
    parser.add_argument("video", type=str, help="Path to the input video file or image sequence pattern.")
    parser.add_argument("--mode", type=str, choices=["mask", "bounding", "path", "dnn"], default="path", help="Operation mode.")
    parser.add_argument("--start", type=int, default=0, help="Starting frame (for path/dnn mode).")
    parser.add_argument("--end", type=int, default=900, help="Ending frame (for path/dnn mode).")
    parser.add_argument("--min-area", type=float, default=500.0, help="Minimum contour area to consider (default: 500).")
    parser.add_argument("--smoothing", type=float, default=0.3, help="Smoothing factor for coordinate update (default: 0.3).")
    # YOLO parameters for dnn mode:
    parser.add_argument("--yolo-cfg", type=str, default="yolov3.cfg", help="Path to YOLO config file.")
    parser.add_argument("--yolo-weights", type=str, default="yolov3.weights", help="Path to YOLO weights file.")
    parser.add_argument("--yolo-names", type=str, default="coco.names", help="Path to file with YOLO class names.")
    args = parser.parse_args()

    motion = MotionTracking(args.video, min_area=args.min_area, smoothing=args.smoothing)
    if args.mode == "mask":
        motion.create_mask()
    elif args.mode == "bounding":
        motion.create_bounding_box()
    elif args.mode == "path":
        motion.create_path(args.start, args.end)
    elif args.mode == "dnn":
        motion.create_dnn_path(args.start, args.end, args.yolo_cfg, args.yolo_weights, args.yolo_names)
    else:
        logger.error("Invalid mode selected.")

if __name__ == '__main__':
    main()
