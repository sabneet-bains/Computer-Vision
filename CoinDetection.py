#!/usr/bin/env python3
"""
Advanced Coin Detection from Video
-----------------------------------
Description:
  This script reads a video file and processes each frame to detect, classify,
  and value coins in real-time. It integrates multiple classical OpenCV techniques:
    - Background subtraction to isolate moving coins.
    - Hough Circle Transform to detect circular coin candidates.
    - Contour analysis (with circularity filtering) to further validate candidates.
  For each candidate, a region of interest (ROI) is extracted and classified using either
  a custom-trained CNN (if available) or a classical heuristic based on coin radius.
  Coins that pass a predefined counting zone are accumulated to compute the total monetary value.
  
Requirements:
  - A video file showing coins moving on a treadmill belt.
  - Optionally, a custom-trained CNN model file (e.g., coin_cnn_model.h5) for coin classification.
  
Usage Example:
  python CoinDetection.py coins_video.mp4 --model coin_cnn_model.h5 --use-cnn True --output annotated_coins.mp4

Author: Sabneet Bains
License: MIT License
"""

import argparse
import logging
import os
import cv2 as cv
import numpy as np
from math import pi, sqrt

# Only import TensorFlow/Keras if CNN mode is enabled.
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoinDetection:
    def __init__(self, video_path: str, model_path: str, use_cnn: bool = True):
        """
        Initialize the coin detection system.

        Parameters:
            video_path (str): Path to the input video file.
            model_path (str): Path to the custom CNN model file.
            use_cnn (bool): Whether to use the CNN for classification (if False, use classical heuristic).
        """
        self.video = cv.VideoCapture(video_path)
        if not self.video.isOpened():
            logger.error("Failed to open video: %s", video_path)
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.frame_width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.total_value = 0.0  # Running total monetary value

        self.use_cnn = use_cnn
        if self.use_cnn:
            if not model_path:
                raise ValueError("CNN mode enabled but model path not provided.")
            logger.info("Using CNN for coin classification.")
            self.model = load_model(model_path)
        else:
            logger.info("Using classical heuristic for coin classification.")

        self.coin_labels = ["Dime", "Penny", "Nickel", "Quarter"]
        self.coin_values = {"Dime": 0.10, "Penny": 0.01, "Nickel": 0.05, "Quarter": 0.25}

        # Define counting zone: coins whose center_x > (frame_width - 100) are counted.
        self.counting_zone = self.frame_width - 100

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply adaptive preprocessing: convert to grayscale, apply CLAHE, and Gaussian blur.

        Parameters:
            frame (np.ndarray): Input BGR frame.

        Returns:
            np.ndarray: Preprocessed grayscale image.
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        blurred = cv.GaussianBlur(gray_eq, (9, 9), 0)
        return blurred

    def background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction to the grayscale frame.

        Parameters:
            frame (np.ndarray): Input BGR frame.

        Returns:
            np.ndarray: Foreground mask.
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Use a simple MOG2 subtractor for background subtraction.
        bg_subtractor = cv.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
        fg_mask = bg_subtractor.apply(gray)
        # Clean up the mask.
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
        return mask_clean

    def detect_coins_combined(self, frame: np.ndarray) -> list:
        """
        Detect coin candidates using both Hough Circle Transform and contour analysis
        on the background-subtracted mask.

        Parameters:
            frame (np.ndarray): Input BGR frame.

        Returns:
            list: List of candidate coins as tuples (x, y, radius).
        """
        candidates = []
        # Preprocess for Hough detection.
        preprocessed = self.preprocess_frame(frame)
        circles = cv.HoughCircles(preprocessed, cv.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                candidates.append((x, y, r))
        # Also perform background subtraction and contour analysis.
        mask = self.background_subtraction(frame)
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv.contourArea(cnt)
            if area < 100:  # ignore small noise
                continue
            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * pi * area / (perimeter * perimeter)
            if circularity > 0.7:  # roughly circular
                (x, y), r = cv.minEnclosingCircle(cnt)
                candidates.append((int(x), int(y), int(r)))
        # Optionally, remove duplicates by merging candidates that are close.
        merged = []
        for cand in candidates:
            if not any(sqrt((cand[0]-m[0])**2 + (cand[1]-m[1])**2) < 10 for m in merged):
                merged.append(cand)
        return merged

    def classify_coin_cnn(self, coin_roi: np.ndarray) -> (str, float):
        """
        Classify the coin ROI using the CNN model.

        Parameters:
            coin_roi (np.ndarray): BGR image of the coin.
            
        Returns:
            tuple: (predicted coin label, confidence score)
        """
        coin_resized = cv.resize(coin_roi, (64, 64))
        coin_normalized = coin_resized.astype('float32') / 255.0
        coin_normalized = np.expand_dims(coin_normalized, axis=0)
        predictions = self.model.predict(coin_normalized)
        label_index = int(np.argmax(predictions))
        confidence = float(predictions[0][label_index])
        return self.coin_labels[label_index], confidence

    def classify_coin_classical(self, radius: int) -> (str, float):
        """
        Classify the coin using a classical heuristic based on the coin's radius.

        Parameters:
            radius (int): Detected coin radius.

        Returns:
            tuple: (predicted coin label, estimated confidence)
        """
        if radius < 32:
            label = "Dime"
        elif 32 <= radius < 38:
            label = "Penny"
        elif 38 <= radius < 44:
            label = "Nickel"
        else:
            label = "Quarter"
        confidence = 0.7  # heuristic confidence
        return label, confidence

    def process_video(self, output_video: str = "annotated_coins.mp4") -> None:
        """
        Process the video frame-by-frame: detect and classify coins, annotate the frames,
        and accumulate the total monetary value. At the end, print the total value.

        Parameters:
            output_video (str): Filename for the annotated output video.
        """
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_video, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        frame_count = 0
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            annotated = frame.copy()
            candidates = self.detect_coins_combined(frame)
            for (x_center, y_center, radius) in candidates:
                # Draw circle for visualization.
                cv.circle(annotated, (x_center, y_center), radius, (0, 255, 0), 2)
                # Define ROI.
                x = max(x_center - radius, 0)
                y = max(y_center - radius, 0)
                w = h = 2 * radius
                coin_roi = frame[y:y+h, x:x+w]
                if coin_roi.size == 0:
                    continue
                # Choose classification method.
                if self.use_cnn:
                    label, conf = self.classify_coin_cnn(coin_roi)
                else:
                    label, conf = self.classify_coin_classical(radius)
                value = self.coin_values.get(label, 0)
                cv.putText(annotated, f"{label} (${value:.2f}, {conf*100:.1f}%)", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Count coin if it passes the counting zone.
                if x_center > self.counting_zone:
                    self.total_value += value
                    cv.circle(annotated, (x_center, y_center), 5, (0, 0, 255), -1)
            writer.write(annotated)
            cv.imshow("Annotated Coins", annotated)
            if cv.waitKey(1) & 0xFF == 27:
                break
            frame_count += 1
        writer.release()
        self.video.release()
        cv.destroyAllWindows()
        logger.info("Processed %d frames.", frame_count)
        logger.info("Total monetary value detected: $%.2f", self.total_value)
        print(f"Total monetary value detected: ${self.total_value:.2f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Live Coin Detection and Classification from Video")
    parser.add_argument("video", type=str, help="Path to the input video file.")
    parser.add_argument("--model", type=str, default="", help="Path to the custom CNN model file (ignored if --use-cnn is False).")
    parser.add_argument("--use-cnn", type=bool, default=True, help="Set to False to use classical heuristic classification.")
    parser.add_argument("--output", type=str, default="annotated_coins.mp4", help="Output video filename.")
    args = parser.parse_args()
    
    if args.use_cnn and not args.model:
        parser.error("--model must be provided when --use-cnn is True.")
    
    coin_detector = CoinDetection(args.video, args.model, use_cnn=args.use_cnn)
    coin_detector.process_video(output_video=args.output)

if __name__ == "__main__":
    main()
