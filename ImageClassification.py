#!/usr/bin/env python3
"""
OpenCV Image Classification
-------------------------------------------------------------------
Description:
    This script extracts various image features for classification purposes.
    Features include histogram statistics, HOG descriptor variance, dominant colors,
    average color, corner variance, edge variance, circle variance, and thumbnail statistics.
    GPU acceleration is used for operations when available. Batch processing and
    parallelization are supported via a command-line flag.

Author: Sabneet Bains
License: MIT License
"""

import os
import sys
import glob
import logging
from typing import List, Union

import cv2 as cv
import numpy as np
import scipy.stats as sp
import pandas as pd

# For parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassification:
    """Image Classification Feature Extractor"""

    def __init__(self, image_path: str) -> None:
        """
        Initialize by loading the image and computing its features.

        Parameters:
            image_path (str): Path to the image file.
        """
        self.image_path: str = image_path
        self.image = cv.imread(image_path)
        if self.image is None:
            logger.error("Failed to load image: %s", image_path)
            raise FileNotFoundError(f"Image not found: {image_path}")

        # GPU availability flag
        self.gpu_available: bool = cv.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            logger.info("GPU acceleration is available.")
        else:
            logger.info("GPU acceleration not available. Using CPU.")

        self.name: str = os.path.basename(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # Initialize features with basic image info
        self.features: List[Union[str, float]] = [self.name, self.height, self.width, self.channels]

        # Compute features
        self.calculate_histograms()
        self.find_dominant_colors()
        self.find_average_color()
        self.calculate_HOG()
        self.find_corners()
        self.find_edges()
        self.find_circles()
        self.calculate_thumbnail()

    def calculate_histograms(self) -> None:
        """Calculates histogram statistics for grayscale and each BGR channel."""
        # Grayscale histogram
        self.grayscale_histogram = cv.calcHist([self.grayscale], [0], None, [256], [0, 256])
        self.grayscale_mean = float(np.mean(self.grayscale_histogram))
        self.grayscale_variance = float(np.var(self.grayscale_histogram))
        self.grayscale_skewness = float(sp.skew(self.grayscale_histogram)[0])
        self.grayscale_kurtosis = float(sp.kurtosis(self.grayscale_histogram)[0])
        self.grayscale_absolute_deviation = float(np.mean(np.abs(self.grayscale_histogram - self.grayscale_mean)))
        self.grayscale_standard_deviation = float(np.std(self.grayscale_histogram))

        # Blue channel histogram
        self.blue_histogram = cv.calcHist([self.image], [0], None, [256], [0, 256])
        self.blue_mean = float(np.mean(self.blue_histogram))
        self.blue_variance = float(np.var(self.blue_histogram))
        self.blue_skewness = float(sp.skew(self.blue_histogram)[0])
        self.blue_kurtosis = float(sp.kurtosis(self.blue_histogram)[0])
        self.blue_absolute_deviation = float(np.mean(np.abs(self.blue_histogram - self.blue_mean)))
        self.blue_standard_deviation = float(np.std(self.blue_histogram))

        # Green channel histogram
        self.green_histogram = cv.calcHist([self.image], [1], None, [256], [0, 256])
        self.green_mean = float(np.mean(self.green_histogram))
        self.green_variance = float(np.var(self.green_histogram))
        self.green_skewness = float(sp.skew(self.green_histogram)[0])
        self.green_kurtosis = float(sp.kurtosis(self.green_histogram)[0])
        self.green_absolute_deviation = float(np.mean(np.abs(self.green_histogram - self.green_mean)))
        self.green_standard_deviation = float(np.std(self.green_histogram))

        # Red channel histogram
        self.red_histogram = cv.calcHist([self.image], [2], None, [256], [0, 256])
        self.red_mean = float(np.mean(self.red_histogram))
        self.red_variance = float(np.var(self.red_histogram))
        self.red_skewness = float(sp.skew(self.red_histogram)[0])
        self.red_kurtosis = float(sp.kurtosis(self.red_histogram)[0])
        self.red_absolute_deviation = float(np.mean(np.abs(self.red_histogram - self.red_mean)))
        self.red_standard_deviation = float(np.std(self.red_histogram))

        histogram_stats = [
            self.grayscale_mean, self.grayscale_variance, self.grayscale_skewness,
            self.grayscale_kurtosis, self.grayscale_absolute_deviation, self.grayscale_standard_deviation,
            self.blue_mean, self.blue_variance, self.blue_skewness, self.blue_kurtosis, self.blue_absolute_deviation, self.blue_standard_deviation,
            self.green_mean, self.green_variance, self.green_skewness, self.green_kurtosis, self.green_absolute_deviation, self.green_standard_deviation,
            self.red_mean, self.red_variance, self.red_skewness, self.red_kurtosis, self.red_absolute_deviation, self.red_standard_deviation
        ]
        self.features.extend(histogram_stats)

    def calculate_HOG(self) -> None:
        """Calculates the HOG descriptor variance and appends it as a feature."""
        # Attempt GPU HOG if available; else, fallback to CPU.
        hog_descriptor = None
        if self.gpu_available and hasattr(cv.cuda, "HOGDescriptor"):
            try:
                hog_gpu = cv.cuda.HOGDescriptor()
                gpu_gray = cv.cuda_GpuMat()
                gpu_gray.upload(self.grayscale)
                hog_descriptor = hog_gpu.compute(gpu_gray).download()
            except Exception as e:
                logger.warning("GPU HOG failed (%s); falling back to CPU HOG.", e)
        if hog_descriptor is None:
            hog = cv.HOGDescriptor()
            hog_descriptor = hog.compute(self.grayscale)
        hog_variance = float(np.var(hog_descriptor))
        self.features.append(hog_variance)

    def find_dominant_colors(self) -> None:
        """Calculates dominant colors via k-means clustering and appends them as features."""
        data = np.float32(self.image.reshape((-1, 3)))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv.kmeans(data, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = centers.ravel().tolist()
        self.features.extend(centers)

    def find_average_color(self) -> None:
        """Calculates the average color of the image and appends it as a feature."""
        avg = np.average(np.average(self.image, axis=0), axis=0)
        self.average_color = avg.tolist()
        self.features.extend(self.average_color)

    def find_corners(self) -> None:
        """Calculates the variance of detected corners and appends it as a feature."""
        if self.gpu_available and hasattr(cv.cuda, "createGoodFeaturesToTrackDetector"):
            try:
                gpu_gray = cv.cuda_GpuMat()
                gpu_gray.upload(self.grayscale)
                detector = cv.cuda.createGoodFeaturesToTrackDetector(self.grayscale.dtype, maxCorners=2, qualityLevel=0.01, minDistance=1)
                corners_gpu = detector.detect(gpu_gray)
                corners = corners_gpu.download() if corners_gpu is not None else None
            except Exception as e:
                logger.warning("GPU corner detection failed (%s); using CPU.", e)
                corners = cv.goodFeaturesToTrack(self.grayscale, 2, 0.01, 1)
        else:
            corners = cv.goodFeaturesToTrack(self.grayscale, 2, 0.01, 1)
        if corners is not None:
            corners = np.int0(corners).ravel()
            self.features.append(float(np.var(corners)))
        else:
            self.features.append(0.0)

    def find_edges(self) -> None:
        """Calculates the variance of edges using Canny and appends it as a feature."""
        # Use GPU accelerated Canny if available.
        if self.gpu_available and hasattr(cv.cuda, "createCannyEdgeDetector"):
            try:
                gpu_gray = cv.cuda_GpuMat()
                gpu_gray.upload(self.grayscale)
                canny_detector = cv.cuda.createCannyEdgeDetector(10, 126)
                gpu_edges = canny_detector.detect(gpu_gray)
                edges = gpu_edges.download()
            except Exception as e:
                logger.warning("GPU Canny failed (%s); using CPU.", e)
                edges = cv.Canny(self.grayscale, 10, 126)
        else:
            edges = cv.Canny(self.grayscale, 10, 126)
        self.features.append(float(np.var(edges)))

    def find_circles(self) -> None:
        """Calculates the variance of detected circles using HoughCircles and appends it as a feature."""
        blurred = cv.medianBlur(self.grayscale, 5)
        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, blurred.shape[0] / 8,
                                  param1=100, param2=30, minRadius=1, maxRadius=30)
        if circles is not None:
            circles = circles.ravel()
            self.features.append(float(np.var(circles)))
        else:
            self.features.append(0.0)

    def calculate_thumbnail(self) -> None:
        """Calculates thumbnail statistics and appends them as features."""
        # Use GPU resize if available.
        if self.gpu_available and hasattr(cv.cuda, "resize"):
            try:
                gpu_image = cv.cuda_GpuMat()
                gpu_image.upload(self.image)
                gpu_thumbnail = cv.cuda.resize(gpu_image, (2, 2), interpolation=cv.INTER_AREA)
                thumbnail = gpu_thumbnail.download()
            except Exception as e:
                logger.warning("GPU resize failed (%s); using CPU.", e)
                thumbnail = cv.resize(self.image, (2, 2), interpolation=cv.INTER_AREA)
        else:
            thumbnail = cv.resize(self.image, (2, 2), interpolation=cv.INTER_AREA)
        self.thumbnail = thumbnail
        self.thumbnail_mean = float(np.mean(thumbnail))
        self.thumbnail_variance = float(np.var(thumbnail))
        self.thumbnail_absolute_deviation = float(np.mean(np.abs(thumbnail - self.thumbnail_mean)))
        self.thumbnail_standard_deviation = float(np.std(thumbnail))
        thumbnail_stats = [
            self.thumbnail_mean, self.thumbnail_variance,
            self.thumbnail_absolute_deviation, self.thumbnail_standard_deviation
        ]
        self.features.extend(thumbnail_stats)

    def write_features(self, output_csv: str) -> None:
        """
        Writes the extracted features to a CSV file.

        Parameters:
            output_csv (str): The CSV file path.
        """
        features_df = pd.DataFrame([self.features])
        features_df.to_csv(output_csv, index=False, header=False, mode='a')
        logger.info("Features written for image: %s", self.name)


def process_image(image_path: str, output_csv: str) -> None:
    """Helper function to process a single image and write features."""
    try:
        classifier = ImageClassification(image_path)
        classifier.write_features(output_csv)
    except Exception as e:
        logger.exception("Failed to process image %s: %s", image_path, e)


def main() -> None:
    """
    Main function to process images for feature extraction.
    Supports batch processing and optional parallelization.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract image classification features from images and save to a CSV file."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--output-csv", type=str, default="features.csv", help="CSV file to append features.")
    parser.add_argument("--num-images", type=int, default=113, help="Number of images to process.")
    parser.add_argument("--parallel", action="store_true", help="Process images in parallel.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_csv = args.output_csv
    num_images = args.num_images

    # Remove output CSV if it exists to start fresh
    if os.path.exists(output_csv):
        os.remove(output_csv)

    # Create a list of image paths (assuming names are "0.jpg", "1.jpg", etc.)
    image_paths = [os.path.join(input_dir, f"{i}.jpg") for i in range(num_images)]

    if args.parallel:
        logger.info("Processing images in parallel...")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, path, output_csv) for path in image_paths]
            for future in as_completed(futures):
                # Exceptions are logged in process_image.
                pass
    else:
        logger.info("Processing images sequentially...")
        for path in image_paths:
            process_image(path, output_csv)


if __name__ == "__main__":
    main()
