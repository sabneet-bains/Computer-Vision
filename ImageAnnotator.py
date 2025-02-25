#!/usr/bin/env python3
"""
OpenCV Image Annotator
----------------------
This script allows users to click on an image to annotate several points,
and then export the annotated points and labels into a CSV file.

Author: Sabneet Bains
License: MIT License
"""

import argparse
import logging
import os
import sys
from typing import Any

import cv2 as cv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAnnotator:
    """Annotates an image with user-provided points and labels."""

    def __init__(self, image_path: str, annotations_csv: str) -> None:
        """
        Initialize the ImageAnnotator by loading the image and setting up filenames.

        Parameters:
            image_path (str): Path to the input image.
            annotations_csv (str): Path to save the annotations CSV file.
        """
        self.image_path: str = image_path
        self.annotations_csv: str = annotations_csv

        # Extract file name and extension using os.path
        base, ext = os.path.splitext(os.path.basename(image_path))
        self.image_name: str = base
        self.image_extension: str = ext.lstrip('.')

        # Load and resize the image for easier viewing
        self.image: np.ndarray = cv.imread(image_path)
        if self.image is None:
            logger.error("Failed to load image from %s", image_path)
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.image = cv.resize(self.image, (1000, 700))

        # Output annotated image filename
        self.annotated_image_filename: str = f"{self.image_name}_annotated.{self.image_extension}"

        # Initialize annotations array with header row
        self.annotations: np.ndarray = np.array([["x", "y", "annotation"]])

    def create_annotation(self, x: int, y: int) -> None:
        """
        Prompt the user for an annotation label, draw the annotation on the image,
        and append the annotation data.

        Parameters:
            x (int): x-coordinate of the annotation.
            y (int): y-coordinate of the annotation.
        """
        annotation: str = input("Please, label this annotation: ")
        new_annotation = np.array([[str(x), str(y), annotation]])
        self.annotations = np.vstack([self.annotations, new_annotation])

        # Draw a small circle and label at the annotation point
        cv.circle(self.image, (x, y), 2, (0, 0, 0), -1)
        cv.putText(self.image, annotation, (x + 5, y + 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def write_annotations(self) -> None:
        """
        Write the annotated image and annotations CSV file to disk.
        """
        cv.imwrite(self.annotated_image_filename, self.image)
        np.savetxt(self.annotations_csv, self.annotations, delimiter=",", fmt="%s")
        logger.info("Annotated image saved as %s", self.annotated_image_filename)
        logger.info("Annotations CSV saved as %s", self.annotations_csv)

    def annotate_image(self) -> None:
        """
        Create an interactive window to annotate the image.
        Left-click to add an annotation, and press 'q' to finish and save.
        """
        cv.namedWindow("Annotate Image", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Annotate Image", self.mouse_callback)

        while True:
            cv.imshow("Annotate Image", self.image)
            key: int = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv.destroyAllWindows()
        # Save annotations after user exits the annotation mode.
        self.write_annotations()

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        Mouse callback to handle annotation creation on left button click.

        Parameters:
            event (int): The mouse event.
            x (int): x-coordinate of the mouse event.
            y (int): y-coordinate of the mouse event.
            flags (int): Any relevant flags passed by OpenCV.
            param (Any): Additional parameters (unused).
        """
        if event == cv.EVENT_LBUTTONDOWN:
            self.create_annotation(x, y)


def main() -> None:
    """
    Parse command-line arguments and run the image annotator.
    """
    parser = argparse.ArgumentParser(description="OpenCV Image Annotator")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("annotations_csv", type=str, help="Path to save the annotations CSV file.")
    args = parser.parse_args()

    try:
        annotator = ImageAnnotator(args.image_path, args.annotations_csv)
        annotator.annotate_image()
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
