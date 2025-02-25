#!/usr/bin/env python3
"""
OpenCV Image Masking - Morphological Transformations (Optimized)
-----------------------------------------------------------------
This script applies various morphological operations to a grayscale image.
GPU acceleration is used when available, and the results for each operation
are collated into a single image table for display or saving.

Author: Sabneet Bains
License: MIT License
"""

import argparse
import logging
import os
from math import ceil
from typing import List, Optional

import cv2 as cv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MorphTransform:
    """Morphological Transformations on a grayscale image."""

    def __init__(self, image_path: str) -> None:
        """
        Initialize the MorphTransform object by loading the grayscale image.

        Parameters:
            image_path (str): Path to the input image.
        """
        self.image = cv.imread(image_path, 0)  # load as grayscale
        if self.image is None:
            logger.error("Failed to load image: %s", image_path)
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.copy = self.image.copy()  # preserve the original copy
        self.eroded: Optional[np.ndarray] = None
        self.dilated: Optional[np.ndarray] = None
        self.opening: Optional[np.ndarray] = None
        self.closing: Optional[np.ndarray] = None
        self.gradient: Optional[np.ndarray] = None
        self.tophat: Optional[np.ndarray] = None
        self.blackhat: Optional[np.ndarray] = None
        self.hitmiss: Optional[np.ndarray] = None

        # Check for GPU acceleration availability
        self.gpu_available: bool = cv.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            logger.info("GPU acceleration is available.")
        else:
            logger.info("GPU acceleration is not available; using CPU.")

    def apply_morphology(self, morph_type: str, kernel: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform the specified morphological operation on the image.

        Parameters:
            morph_type (str): One of 'erosion', 'dilation', 'opening', 'closing',
                              'gradient', 'tophat', 'blackhat', or 'hitmiss'.
            kernel (np.ndarray): The structuring element (kernel) to use.

        Returns:
            Optional[np.ndarray]: The resulting image after the operation.
        """
        # Map string type to the corresponding OpenCV morphological flag.
        morph_flag = {
            'erosion': cv.MORPH_ERODE,
            'dilation': cv.MORPH_DILATE,
            'opening': cv.MORPH_OPEN,
            'closing': cv.MORPH_CLOSE,
            'gradient': cv.MORPH_GRADIENT,
            'tophat': cv.MORPH_TOPHAT,
            'blackhat': cv.MORPH_BLACKHAT,
            'hitmiss': cv.MORPH_HITMISS
        }.get(morph_type, None)

        if morph_flag is None:
            logger.error("Invalid morphological operation: %s", morph_type)
            return None

        # Try GPU-accelerated processing if available.
        if self.gpu_available:
            try:
                gpu_src = cv.cuda_GpuMat()
                gpu_src.upload(self.copy)
                # Create a morphology filter using the specified flag.
                morph_filter = cv.cuda.createMorphologyFilter(morph_flag, self.copy.dtype, kernel)
                result_gpu = morph_filter.apply(gpu_src)
                result = result_gpu.download()
                logger.info("GPU %s succeeded.", morph_type)
            except Exception as e:
                logger.warning("GPU operation for %s failed: %s. Falling back to CPU.", morph_type, e)
                result = cv.morphologyEx(self.copy, morph_flag, kernel)
        else:
            result = cv.morphologyEx(self.copy, morph_flag, kernel)

        # Save the result in the appropriate member variable.
        if morph_type == 'erosion':
            self.eroded = result
        elif morph_type == 'dilation':
            self.dilated = result
        elif morph_type == 'opening':
            self.opening = result
        elif morph_type == 'closing':
            self.closing = result
        elif morph_type == 'gradient':
            self.gradient = result
        elif morph_type == 'tophat':
            self.tophat = result
        elif morph_type == 'blackhat':
            self.blackhat = result
        elif morph_type == 'hitmiss':
            self.hitmiss = result

        return result

    def write_image(self, images: List[np.ndarray], output_path: str) -> None:
        """
        Create an image table from a list of images and write it to a file.

        Parameters:
            images (List[np.ndarray]): List of images to include in the table.
            output_path (str): File path for the output image.
        """
        n = len(images)
        if n == 0:
            logger.error("No images to write.")
            return
        split = int(ceil(n / 2))
        top_row = np.hstack(images[:split])
        if n > split:
            bottom_row = np.hstack(images[split:])
            table = np.vstack([top_row, bottom_row])
        else:
            table = top_row
        cv.imwrite(output_path, table)
        logger.info("Image table written to %s", output_path)

    def show_image(self, images: List[np.ndarray]) -> None:
        """
        Create an image table from a list of images and display it in a window.

        Parameters:
            images (List[np.ndarray]): List of images to display.
        """
        n = len(images)
        if n == 0:
            logger.error("No images to display.")
            return
        split = int(ceil(n / 2))
        top_row = np.hstack(images[:split])
        if n > split:
            bottom_row = np.hstack(images[split:])
            table = np.vstack([top_row, bottom_row])
        else:
            table = top_row
        cv.imshow('Morphological Transformations', table)
        cv.waitKey(0)
        cv.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply morphological transformations to a grayscale image and display or save the results."
    )
    parser.add_argument("image", type=str, help="Path to the input image.")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size (default: 5)")
    parser.add_argument("--operations", type=str, nargs='*', default=[
        'erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat', 'hitmiss'
    ], help="List of morphological operations to apply.")
    parser.add_argument("--output", type=str, default="morphology.png", help="Output file for the image table.")
    parser.add_argument("--display", action="store_true", help="Display the image table.")
    args = parser.parse_args()

    morph = MorphTransform(args.image)
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    results = []

    for op in args.operations:
        res = morph.apply_morphology(op, kernel)
        if res is not None:
            results.append(res)
        else:
            logger.warning("Skipping invalid operation: %s", op)

    if results:
        morph.write_image(results, args.output)
        if args.display:
            morph.show_image(results)
    else:
        logger.error("No valid morphological operations were applied.")


if __name__ == "__main__":
    main()
