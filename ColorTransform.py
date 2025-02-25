#!/usr/bin/env python3
"""
Module: color_transform
Description:
    This module performs color model transformations on an input image,
    optimized for speed. It leverages GPU acceleration (if available) to
    perform HSV extraction and hue shifting, falling back to CPU processing
    if no GPU is present. The module then creates a collage showing the
    original image alongside the hue-shifted image.

Author: Sabneet Bains
License: MIT License
"""

import argparse
import logging
import sys
from typing import List, Tuple

import cv2 as cv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_image_collage(images: List[np.ndarray], labels: List[str]) -> np.ndarray:
    """
    Create a collage from a list of images with corresponding labels.

    For two images, they are concatenated horizontally. For more images,
    a two-row grid is created.

    Parameters:
        images (List[np.ndarray]): List of images (assumed to have the same dimensions).
        labels (List[str]): List of labels corresponding to each image.

    Returns:
        np.ndarray: The collage image with labels.
    """
    if len(images) != len(labels):
        raise ValueError("The number of images and labels must match.")

    labeled_images = []
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 255, 0)  # Green text

    for img, label in zip(images, labels):
        # If the image is grayscale, convert it to BGR for colored text overlay.
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()
        cv.putText(img_bgr, label, (10, 30), font, font_scale, color, thickness)
        labeled_images.append(img_bgr)

    if len(labeled_images) == 2:
        collage = cv.hconcat(labeled_images)
    else:
        split = int(np.ceil(len(labeled_images) / 2))
        top_row = cv.hconcat(labeled_images[:split])
        bottom_row = cv.hconcat(labeled_images[split:])
        collage = cv.vconcat([top_row, bottom_row])
    return collage


class ColorTransform:
    """
    Color Model Transformations with optional GPU acceleration.
    """

    def __init__(self, image_path: str) -> None:
        """
        Initialize by loading the image and checking for GPU support.

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
        self.shifted_image = None

    def extract_hsv_cpu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        return cv.split(hsv)

    def extract_hsv_gpu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Upload image to GPU and convert to HSV
        gpu_image = cv.cuda_GpuMat()
        gpu_image.upload(self.image)
        gpu_hsv = cv.cuda.cvtColor(gpu_image, cv.COLOR_BGR2HSV)
        # Download the HSV image to CPU for channel operations
        hsv = gpu_hsv.download()
        return cv.split(hsv)

    def shift_hue(self, hue_shift: int) -> np.ndarray:
        """
        Shift the hue of the image by hue_shift degrees. Hue values wrap modulo 180.

        Parameters:
            hue_shift (int): Value to shift the hue.

        Returns:
            np.ndarray: The BGR image after hue shifting.
        """
        # Use GPU or CPU extraction depending on availability.
        if self.gpu_available:
            logger.info("Using GPU for hue extraction.")
            h, s, v = self.extract_hsv_gpu()
        else:
            logger.info("Using CPU for hue extraction.")
            h, s, v = self.extract_hsv_cpu()

        # Shift hue with modular arithmetic.
        h_shifted = ((h.astype(np.int32) + hue_shift) % 180).astype(np.uint8)
        hsv_shifted = cv.merge((h_shifted, s, v))

        # Convert back to BGR using GPU if available.
        if self.gpu_available:
            gpu_hsv_shifted = cv.cuda_GpuMat()
            gpu_hsv_shifted.upload(hsv_shifted)
            gpu_bgr = cv.cuda.cvtColor(gpu_hsv_shifted, cv.COLOR_HSV2BGR)
            shifted = gpu_bgr.download()
        else:
            shifted = cv.cvtColor(hsv_shifted, cv.COLOR_HSV2BGR)
        self.shifted_image = shifted
        return shifted

    def show_collage(self, shifted: np.ndarray) -> None:
        """
        Create and display a collage with the original and hue-shifted images.
        """
        collage = create_image_collage([self.image, shifted], ["Original", "Hue Shifted"])
        cv.imshow("Color Transformation Collage", collage)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save_collage(self, shifted: np.ndarray, output_path: str) -> None:
        """
        Create a collage with the original and hue-shifted images and save it.

        Parameters:
            shifted (np.ndarray): Hue-shifted image.
            output_path (str): File path to save the collage.
        """
        collage = create_image_collage([self.image, shifted], ["Original", "Hue Shifted"])
        cv.imwrite(output_path, collage)
        logger.info("Collage saved to %s", output_path)


def main() -> None:
    """
    Parse command-line arguments, perform the hue shift, and display/save the collage.
    """
    parser = argparse.ArgumentParser(
        description="Optimized Color Transformation with optional GPU acceleration."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--hue-shift", type=int, default=110, help="Hue shift value (default: 110).")
    parser.add_argument("--output", type=str, default="output_collage.jpg", help="Path to save the collage.")
    args = parser.parse_args()

    try:
        transformer = ColorTransform(args.image_path)
        shifted = transformer.shift_hue(args.hue_shift)
        transformer.show_collage(shifted)
        transformer.save_collage(shifted, args.output)
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
