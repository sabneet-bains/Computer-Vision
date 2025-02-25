#!/usr/bin/env python3
"""
Module: discrete_fourier_transform
This module performs the Discrete Fourier Transform (DFT) and its inverse (IDFT)
on a grayscale image. It checks for GPU acceleration; if available, it uses
OpenCV's CUDA functions; otherwise, it falls back to CPU processing.

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


class DiscreteFourierTransform:
    """
    Class for computing the DFT and inverse DFT (IDFT) on a grayscale image.
    Uses GPU acceleration if available.
    """

    def __init__(self, image_path: str) -> None:
        """
        Initialize the processor by loading the image and checking for GPU availability.

        Parameters:
            image_path (str): Path to the input image file.

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        self.image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if self.image is None:
            logger.error("Image not found or unable to read: %s", image_path)
            raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

        # Check for GPU support
        self.use_gpu: bool = cv.cuda.getCudaEnabledDeviceCount() > 0
        logger.info("GPU acceleration available: %s", self.use_gpu)

        # To store results
        self.magnitude: Optional[np.ndarray] = None
        self.identity: Optional[np.ndarray] = None

    def process(self) -> None:
        """
        Process the image using either GPU or CPU based on availability.
        Computes both the magnitude spectrum and the reconstructed image.
        """
        if self.use_gpu:
            self._process_gpu()
        else:
            self._process_cpu()

    def _process_gpu(self) -> None:
        """
        Process the image using GPU-accelerated DFT and IDFT.
        """
        logger.info("Processing using GPU acceleration...")

        # Convert image to float32 and upload to GPU
        gpu_image = cv.cuda_GpuMat()
        gpu_image.upload(np.float32(self.image))

        # Compute the DFT on GPU with complex output
        gpu_dft = cv.cuda.dft(gpu_image, flags=cv.DFT_COMPLEX_OUTPUT)

        # Download DFT result to CPU for magnitude calculation
        dft = gpu_dft.download()
        # Shift zero-frequency component to center for display purposes
        dft_shift = np.fft.fftshift(dft)
        # Compute magnitude and phase (only magnitude is used for display)
        magnitude, _ = cv.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
        # Normalize magnitude for display
        self.magnitude = np.uint8(cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX))

        # For reconstruction, perform the inverse DFT on GPU directly.
        gpu_idft = cv.cuda.idft(gpu_dft, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        idft = gpu_idft.download()
        # Normalize the reconstructed image to the range [0, 255]
        self.identity = np.uint8(cv.normalize(idft, None, 0, 255, cv.NORM_MINMAX))

        logger.info("GPU processing completed successfully.")

    def _process_cpu(self) -> None:
        """
        Process the image using CPU-based DFT and IDFT.
        """
        logger.info("Processing using CPU...")

        image_float = np.float32(self.image)
        # Compute DFT with complex output
        dft = cv.dft(image_float, flags=cv.DFT_COMPLEX_OUTPUT)

        # Shift zero-frequency component to the center for visualization
        dft_shift = np.fft.fftshift(dft)
        magnitude, _ = cv.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
        self.magnitude = np.uint8(cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX))

        # For reconstruction, reverse the FFT shift and compute the inverse DFT
        idft_shift = np.fft.ifftshift(dft_shift)
        idft = cv.idft(idft_shift)
        idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
        self.identity = np.uint8(cv.normalize(idft_magnitude, None, 0, 255, cv.NORM_MINMAX))

        logger.info("CPU processing completed successfully.")

    def save_and_show(self, original_path: str, magnitude_path: str, identity_path: str) -> None:
        """
        Save the original, magnitude, and reconstructed images, and display them as a single collage.
        The collage image displays all three images side by side with labeled descriptions.

        Parameters:
            original_path (str): Output path for the original image.
            magnitude_path (str): Output path for the magnitude spectrum image.
            identity_path (str): Output path for the reconstructed image.
        """
        # Save individual images to disk
        cv.imwrite(original_path, self.image)
        cv.imwrite(magnitude_path, self.magnitude)
        cv.imwrite(identity_path, self.identity)
        logger.info("Images saved:\nOriginal: %s\nMagnitude: %s\nReconstructed: %s",
                    original_path, magnitude_path, identity_path)

        # Convert grayscale images to BGR for color labeling
        original_bgr = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
        magnitude_bgr = cv.cvtColor(self.magnitude, cv.COLOR_GRAY2BGR)
        identity_bgr = cv.cvtColor(self.identity, cv.COLOR_GRAY2BGR)

        # Define label properties
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (0, 255, 0)  # Green text
        label_positions = {
            "Original": (10, 30),
            "Magnitude": (10, 30),
            "Identity": (10, 30),
        }

        # Add labels to each image
        cv.putText(original_bgr, "Original", label_positions["Original"], font, font_scale, color, thickness)
        cv.putText(magnitude_bgr, "Magnitude", label_positions["Magnitude"], font, font_scale, color, thickness)
        cv.putText(identity_bgr, "Identity", label_positions["Identity"], font, font_scale, color, thickness)

        # Concatenate images horizontally to create a collage
        collage = cv.hconcat([original_bgr, magnitude_bgr, identity_bgr])

        # Save the collage image
        collage_path = "collage.jpg"
        cv.imwrite(collage_path, collage)
        logger.info("Collage image saved to %s", collage_path)

        # Display the collage in a single window
        cv.imshow("DFT Processing Collage", collage)
        cv.waitKey(0)
        cv.destroyAllWindows()


def main() -> None:
    """
    Parse command-line arguments and execute the DFT/IDFT processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Apply DFT and IDFT on a grayscale image with optional GPU acceleration."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--output-original", type=str, default="Original.jpg",
                        help="Output path for the original image.")
    parser.add_argument("--output-magnitude", type=str, default="Magnitude.jpg",
                        help="Output path for the magnitude spectrum image.")
    parser.add_argument("--output-identity", type=str, default="Identity.jpg",
                        help="Output path for the reconstructed image.")
    args = parser.parse_args()

    try:
        processor = DiscreteFourierTransform(args.image_path)
        processor.process()
        processor.save_and_show(args.output_original, args.output_magnitude, args.output_identity)
    except Exception as e:
        logger.exception("An error occurred during processing: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
