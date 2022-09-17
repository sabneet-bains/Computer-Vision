''' OpenCV Image Annotator '''
# Python 3.10.4
# Author: Sabneet Bains
# Date: 06-05-2022
# Version: 0.1
# Description: This script allows users to click on an image,
             # annotate several points within the image, and export the
             # annotated points and labels into a CSV file.
import sys
import numpy as np
import cv2 as cv

class ImageAnnotator():
    ''' Annotates an image object '''

    # constructor
    def __init__(self, image_path, annotations_csv):
        self.image_path = image_path # for now assumes the same directory as the script
        self.image_name = self.image_path.split('.')[0]
        self.image_extension = self.image_path.split('.')[-1]
        self.image = cv.imread(self.image_path)
        self.image = cv.resize(self.image, (1000, 700)) # for easier viewing
        self.annotated_image = self.image_name + '_annotated.' + self.image_extension
        self.annotations = np.array(['x', 'y', 'annotation'])
        self.annotations_csv = annotations_csv

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

    def get_annotated_image(self):
        ''' Returns the annotated image filename '''
        return self.annotated_image

    def get_annotations(self):
        ''' Returns the annotations '''
        return self.annotations

    def get_annotations_csv(self):
        ''' Returns the annotations CSV file '''
        return self.annotations_csv

    # setters
    def set_image_path(self, image_path):
        ''' Sets the image path '''
        self.image_path = image_path

    def set_annotations_csv(self, annotations_csv):
        ''' Sets the annotations CSV file '''
        self.annotations_csv = annotations_csv


    # methods
    def create_annotations(self, x, y):
        ''' Creates the annotations '''

        annotation = input("Please, label this annotation: ")
        self.annotations = np.vstack([self.annotations, [str(x), str(y), str(annotation)]])

        cv.circle(self.image, (x, y), 2, (0, 0, 0), -1)
        cv.putText(self.image, annotation, (x + 5, y + 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    def write_annotations(self):
        ''' Writes the annotations to an output image and a CSV file '''

        cv.imwrite(self.annotated_image, self.image)
        np.savetxt(self.annotations_csv, self.annotations, delimiter=',', fmt='%s')

    def annotate_image(self):
        ''' Creates a window to annotate the image '''

        cv.namedWindow("Annotate Image", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Annotate Image", self.mouse_callback)

        while True:
            cv.imshow("Annotate Image", self.image)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        ''' Mouse callback function '''

        if event == cv.EVENT_LBUTTONDOWN:
            self.create_annotations(x, y)
            self.write_annotations()


# main
if __name__ == "__main__":
    # create an image annotator
    fruits = ImageAnnotator(sys.argv[1], sys.argv[2])
    fruits.annotate_image()
    