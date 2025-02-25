#!/usr/bin/env python3
"""
Module: edge_detection
Description:
    This module performs edge detection on an input image. It first converts
    the image to grayscale, applies a Gaussian blur, and then uses Canny edge
    detection to extract edges. If a GPU is available, the CUDA-accelerated
    functions are used for faster processing; otherwise, the CPU pipeline is used.

Author: Sabneet Bains
License: MIT License
"""

import argparse
import logging
import sys
from typing import Optional

import cv2 as cv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeDetection:
    """
    Class for performing edge detection on an image, with optional GPU acceleration.
    """

    def __init__(self, image_path: str) -> None:
        """
        Initialize the EdgeDetection object by loading the image and checking GPU support.

        Parameters:
            image_path (str): Path to the input image.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """
        self.image = cv.imread(image_path)
        if self.image is None:
            logger.error("Image not found or unable to load: %s", image_path)
            raise FileNotFoundError(f"Image not found or unable to load: {image_path}")

        # Check for GPU support
        self.gpu_available: bool = cv.cuda.getCudaEnabledDeviceCount() > 0
        logger.info("GPU acceleration available: %s", self.gpu_available)
        self.result: Optional[np.ndarray] = None

    def process_cpu(self, kernel_size: int, threshold1: int, threshold2: int) -> np.ndarray:
        """
        Process the image using CPU-based methods.

        Parameters:
            kernel_size (int): Size of the kernel for Gaussian blur.
            threshold1 (int): First threshold for the hysteresis procedure in Canny.
            threshold2 (int): Second threshold for the hysteresis procedure in Canny.

        Returns:
            np.ndarray: Edge-detected image.
        """
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        edges = cv.Canny(blur, threshold1, threshold2)
        return edges

    def process_gpu(self, kernel_size: int, threshold1: int, threshold2: int) -> np.ndarray:
        """
        Process the image using GPU-accelerated methods.

        Parameters:
            kernel_size (int): Size of the kernel for Gaussian blur.
            threshold1 (int): First threshold for the hysteresis procedure in Canny.
            threshold2 (int): Second threshold for the hysteresis procedure in Canny.

        Returns:
            np.ndarray: Edge-detected image.
        """
        # Upload image to GPU and convert to grayscale
        gpu_image = cv.cuda_GpuMat()
        gpu_image.upload(self.image)
        gpu_gray = cv.cuda.cvtColor(gpu_image, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur using a CUDA filter
        gaussian_filter = cv.cuda.createGaussianFilter(gpu_gray.type(), gpu_gray.type(), (kernel_size, kernel_size), 0)
        gpu_blur = gaussian_filter.apply(gpu_gray)

        # Apply Canny edge detection using CUDA
        canny_detector = cv.cuda.createCannyEdgeDetector(threshold1, threshold2)
        gpu_edges = canny_detector.detect(gpu_blur)

        # Download the result back to CPU memory
        edges = gpu_edges.download()
        return edges

    def process(self, kernel_size: int, threshold1: int, threshold2: int) -> np.ndarray:
        """
        Process the image using either GPU or CPU pipeline based on availability.

        Parameters:
            kernel_size (int): Size of the kernel for Gaussian blur.
            threshold1 (int): First threshold for Canny.
            threshold2 (int): Second threshold for Canny.

        Returns:
            np.ndarray: Edge-detected image.
        """
        if self.gpu_available:
            logger.info("Using GPU acceleration for edge detection.")
            self.result = self.process_gpu(kernel_size, threshold1, threshold2)
        else:
            logger.info("Using CPU for edge detection.")
            self.result = self.process_cpu(kernel_size, threshold1, threshold2)
        return self.result

    def show_image(self) -> None:
        """
        Display the edge-detected image.
        """
        if self.result is None:
            raise ValueError("No result to display. Call process() first.")
        cv.imshow("Edge Detection Result", self.result)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def write_image(self, output_path: str) -> None:
        """
        Write the edge-detected image to a file.

        Parameters:
            output_path (str): File path to save the image.
        """
        if self.result is None:
            raise ValueError("No result to write. Call process() first.")
        cv.imwrite(output_path, self.result)
        logger.info("Edge-detected image saved to %s", output_path)


def main() -> None:
    """
    Parse command-line arguments, perform edge detection, and show/save the result.
    """
    parser = argparse.ArgumentParser(
        description="Perform edge detection on an image with optional GPU acceleration."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size for Gaussian blur (default: 5).")
    parser.add_argument("--threshold1", type=int, default=100, help="First threshold for Canny (default: 100).")
    parser.add_argument("--threshold2", type=int, default=200, help="Second threshold for Canny (default: 200).")
    parser.add_argument("--output", type=str, default="EdgeDetected.jpg", help="Path to save the output image.")
    parser.add_argument("--display", action="store_true", help="Display the output image in a window.")
    args = parser.parse_args()

    try:
        detector = EdgeDetection(args.image_path)
        detector.process(args.kernel_size, args.threshold1, args.threshold2)
        detector.write_image(args.output)
        if args.display:
            detector.show_image()
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
