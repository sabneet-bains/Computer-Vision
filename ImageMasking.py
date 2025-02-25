#!/usr/bin/env python3
"""
OpenCV Image Masking (Optimized)
--------------------------------
Usage:
  python project_02.py [--image fruits.jpg]

Keys:
  r     - mask the image
  SPACE - reset the inpainting mask
  ESC   - exit

Description:
  This script allows users to apply non-destructive masks by highlighting 
  specific objects in an image, then outputs the masked image and an image 
  table of input, mask, and output.

Author: Sabneet Bains
License: MIT License
"""

import argparse
import logging
import os
import sys
import numpy as np
import cv2 as cv

from common import Sketcher  # Assumes Sketcher is available in common.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageMasking:
    """Applies a mask to an image using interactive sketching."""

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        base, ext = os.path.splitext(os.path.basename(image_path))
        self.image_name = base
        self.image_extension = ext.lstrip('.')
        self.image = cv.imread(image_path)
        if self.image is None:
            logger.error("Failed to load image: %s", image_path)
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.masked_image = f"{self.image_name}_masked.{self.image_extension}"
        self.masked_image_table = f"{self.image_name}_masked_table.{self.image_extension}"
        # Check for GPU availability
        self.gpu_available = cv.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            logger.info("GPU acceleration available.")
        else:
            logger.info("GPU acceleration not available; using CPU.")

    def create_mask(self, image_mark: np.ndarray) -> None:
        """Creates a mask on the image from the user sketch."""
        # Create a mask from the sketch. Try GPU inRange if available (else fallback to CPU)
        try:
            if self.gpu_available:
                # Note: cv.cuda.inRange is not exposed in Python; we fall back to CPU.
                mask = cv.inRange(image_mark, np.array([0, 0, 255]), np.array([255, 255, 255]))
            else:
                mask = cv.inRange(image_mark, np.array([0, 0, 255]), np.array([255, 255, 255]))
        except Exception as e:
            logger.warning("GPU accelerated inRange failed: %s. Falling back to CPU.", e)
            mask = cv.inRange(image_mark, np.array([0, 0, 255]), np.array([255, 255, 255]))

        mask_inv = cv.bitwise_not(mask)

        # Convert the original image to grayscale (using GPU if possible)
        try:
            if self.gpu_available:
                gpu_img = cv.cuda_GpuMat()
                gpu_img.upload(self.image)
                gpu_gray = cv.cuda.cvtColor(gpu_img, cv.COLOR_BGR2GRAY)
                img2gray = gpu_gray.download()
            else:
                img2gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        except Exception as e:
            logger.warning("GPU accelerated cvtColor failed: %s. Using CPU.", e)
            img2gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        rows, cols = self.image.shape[:2]

        # Apply the mask to get the colored region
        color = cv.bitwise_or(self.image, self.image, mask=mask)
        # Apply the inverse mask to the grayscale image
        grayscale = cv.bitwise_or(img2gray, img2gray, mask=mask_inv)
        # Convert grayscale to 3-channel image for combination
        grayscale = cv.cvtColor(grayscale, cv.COLOR_GRAY2BGR)

        # Write the masked image and the image table
        self.write_masked_image(color, grayscale, mask)

    def write_masked_image(self, color: np.ndarray, grayscale: np.ndarray, mask: np.ndarray) -> None:
        """Writes the masked image and a combined image table to files."""
        # Convert the mask to a 3-channel image using cvtColor (more efficient than np.stack)
        mask_color = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        mask_output = cv.add(color, grayscale)
        table_output = np.concatenate((self.image, mask_color, mask_output), axis=1)

        cv.imwrite(self.masked_image, mask_output)
        cv.imwrite(self.masked_image_table, table_output)
        cv.imshow('Image Table', table_output)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def sketch_mask(self) -> None:
        """Creates an interactive window to sketch the mask."""
        image_mark = self.image.copy()
        mark = np.zeros(self.image.shape[:2], np.uint8)  # Black mask
        sketch = Sketcher('image', [image_mark, mark], lambda: ((255, 255, 255), 255))

        while True:
            key = cv.waitKey()
            if key == 27:  # ESC key to exit
                break
            if key == ord('r'):
                self.create_mask(image_mark)
                break
            if key == ord(' '):
                image_mark[:] = self.image
                sketch.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV Image Masking")
    parser.add_argument("--image", type=str, default="fruits.jpg", help="Path to the input image.")
    args = parser.parse_args()

    logger.info("Starting image masking for image: %s", args.image)
    masker = ImageMasking(args.image)
    masker.sketch_mask()


if __name__ == "__main__":
    main()
