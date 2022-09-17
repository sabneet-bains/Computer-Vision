''' OpenCV Image Masking:

Usage:
  project_02.py [<image>]

Keys:
  r     - mask the image
  SPACE - reset the inpainting mask
  ESC   - exit
'''
# Python 3.10.5
# Author: Sabneet Bains
# Date: 06-20-2022
# Version: 0.1
# Description: This script allows users to apply non-destructive masks by
             # highlighting specific objects in an image, and then outputs
             # the masked image and an image table of input, mask, and output.

import sys
import numpy as np
import cv2 as cv

from common import Sketcher


class ImageMasking():
    ''' Applies a mask to an image '''

    # constructor
    def __init__(self, image_path):
        self.image_path = image_path # for now assumes the same directory as the script
        self.image_name = self.image_path.split('.')[0]
        self.image_extension = self.image_path.split('.')[-1]
        self.image = cv.imread(self.image_path)
        self.masked_image = self.image_name + '_masked.' + self.image_extension
        self.masked_image_table = self.image_name + '_masked_table.' + self.image_extension

    # getters
    def get_image_path(self):
        ''' Returns the image path '''
        return self.image_path

    def get_image_name(self):
        ''' Returns the image name '''
        return self.image_name

    def get_image_extension(self):
        ''' Returns the image extension '''
        return self.image_extension

    def get_original_image(self):
        ''' Returns the original image '''
        return self.image

    def get_masked_image(self):
        ''' Returns the masked image filename '''
        return self.masked_image

    def get_masked_image_table(self):
        ''' Returns the masked image table filename '''
        return self.masked_image_table

    # setters
    def set_image_path(self, image_path):
        ''' Sets the image path '''
        self.image_path = image_path

    # methods
    def create_mask(self, image_mark):
        ''' Creates a mask on the image '''

        # create a mask and an inverse from the sketch
        mask = cv.inRange(image_mark, np.array([0,0,255]), np.array([255,255,255]))
        mask_inv = cv.bitwise_not(mask)

        # create a grayscale image from the original image
        img2gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        # get dimensions of the original image
        rows, cols, _ = self.image.shape
        self.image = self.image[0:rows, 0:cols]

        # apply the mask to the image
        color = cv.bitwise_or(self.image, self.image, mask = mask)
        color = color[0:rows, 0:cols]

        # apply the inverse mask to the grayscale image
        grayscale = cv.bitwise_or(img2gray, img2gray, mask = mask_inv)
        grayscale = np.stack((grayscale,)*3, axis=-1)

        # write the masked image to a file
        self.write_masked_image(color, grayscale, mask)

    def write_masked_image(self, color, grayscale, mask):
        ''' Writes the masked image to a file '''

        # convert to 3 channel image
        mask = np.stack((mask,)*3, axis=-1)

        # add the color image to the grayscale image
        mask_output = cv.add(color, grayscale)
        table_output = np.concatenate((self.image, mask, mask_output), axis=1)

        # write the masked image and image table to files
        cv.imwrite(self.masked_image, mask_output)
        cv.imwrite(self.masked_image_table, table_output)
        cv.imshow('Image Table', table_output)
        cv.waitKey(0) # Wait for a keyboard event

    def sketch_mask(self):
        ''' Sketches a mask on the image '''
        image_mark = self.image.copy()
        mark = np.zeros(self.image.shape[:2], np.uint8) # create a black mask
        sketch = Sketcher('image', [image_mark, mark], lambda : ((255, 255, 255), 255)) # create a sketcher

        # Wait until the user is done sketching
        while True:
            key = cv.waitKey()
            if key == 27:
                break
            if key == ord('r'):
                self.create_mask(image_mark)
                break
            if key == ord(' '):
                image_mark[:] = self.image
                sketch.show()

if __name__ == "__main__":
    print(__doc__)
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = 'fruits.jpg'
    fruits = ImageMasking(filename)
    fruits.sketch_mask()
