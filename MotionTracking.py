''' OpenCV Image Annotator '''
# Python 3.10.5
# Author: Sabneet Bains
# Date: 07-16-2022
# Version: 0.1
# Description: This script allows users to perform video motion tracking to detect
             # and track moving objects within a video or sequence of images
import numpy as np
import cv2 as cv

class MotionTracking():
    ''' Motion tracking class '''

    # constructor
    def __init__(self, video_path):
        self.video = cv.VideoCapture(video_path)
        self.background = cv.createBackgroundSubtractorMOG2()

        self.frame_width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.video.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv.CAP_PROP_FPS)

        self.kernel = None
        self.frame = None
        self.mask = None

        self.contours = None
        self.areas = None
        self.max_contour = None

        self.masked = None
        self.bounded = None
        self.tracked = None
        self.starting_x = None
        self.starting_y = None

    # methods
    def preprocess(self, morph_size, blur_size):
        ''' Applies morphological operations and blurring to the frame '''
        _, self.frame = self.video.read()
        self.mask = self.background.apply(self.frame)
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_size, morph_size))
        self.mask = cv.morphologyEx(self.mask, cv.MORPH_OPEN, self.kernel)
        self.mask = cv.medianBlur(self.mask, blur_size)

    def find_contours(self):
        ''' Finds contours in the mask '''
        self.contours, _ = cv.findContours(self.mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
        self.areas = [cv.contourArea(c) for c in self.contours]

    def check_contours(self):
        ''' Checks contour sizes '''
        self.max_contour = self.contours[np.argmax(self.areas)]
        self.x, self.y, self.w, self.h = cv.boundingRect(self.max_contour)

    def create_mask(self):
        ''' Creates mask for background subtraction '''
        self.masked = cv.VideoWriter('masked.mp4', -1, self.fps, (self.frame_width, self.frame_height))

        for _ in range(self.frame_count):
            self.preprocess(morph_size=5, blur_size=5)
            self.masked.write(self.mask)

    def create_bounding_box(self):
        ''' Creates bounding boxes for the objects '''
        self.bounded = cv.VideoWriter('bounded.mp4', -1, self.fps, (self.frame_width, self.frame_height))

        for _ in range(self.frame_count):
            self.preprocess(morph_size=3, blur_size=5)
            self.find_contours()

            if len(self.areas) > 0:
                self.check_contours()
                cv.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)

            self.bounded.write(self.frame)

    def create_path(self, starting_frame, ending_frame):
        ''' Creates a path for an object '''
        self.tracked = cv.VideoWriter('tracked.mp4', -1, self.fps, (self.frame_width, self.frame_height))

        for _ in range(starting_frame, ending_frame):
            self.preprocess(morph_size=5, blur_size=11)
            self.find_contours()

            if len(self.areas) > 1:
                self.check_contours()

                if self.starting_x is None and self.starting_y is None:
                    self.starting_x = self.x
                    self.starting_y = self.y

                cv.line(self.frame, (self.starting_x, self.starting_y), (self.x, self.y), (0, 0, 255), 2)

            self.tracked.write(self.frame)
        cv.imwrite('tracked.jpg', self.frame)

# main
if __name__ == '__main__':
    ippr = MotionTracking('WalkByShop1front%04d.jpg')
    # ippr.create_mask()
    # ippr.create_bounding_box()
    ippr.create_path(0, 900)
