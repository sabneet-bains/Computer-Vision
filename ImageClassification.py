''' OpenCV Image Classification '''
# Python 3.10.6
# Author: Sabneet Bains
# Date: 08-14-2022
# Version: 0.1
# Description: This script implements and exercise some of the basic
             # concepts of image classification using OpenCV.
import cv2 as cv
import numpy as np
import scipy.stats as sp
import pandas as pd

class ImageClassification():
    ''' Image Classification '''

    # constructor
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.name = image_path.split('\\')[-1]
        self.height, self.width, self.channels = np.shape(self.image)
        self.grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.features = [self.name, self.height, self.width, self.channels]
        self.calculate_histograms()
        self.find_dominant_colors()
        self.find_average_color()
        self.calculate_HOG()
        self.find_corners()
        self.find_edges()
        self.find_circles()
        self.calculate_thumbnail()

    # methods
    def calculate_histograms(self):
        ''' Calculates histogram statistics for each channel '''
        self.grayscale_histogram = cv.calcHist([self.grayscale], [0], None, [256], [0, 256])
        self.grayscale_mean = np.mean(self.grayscale_histogram)
        self.grayscale_variance = np.var(self.grayscale_histogram)
        self.grayscale_skewness = sp.skew(self.grayscale_histogram)
        self.grayscale_kurtosis = sp.kurtosis(self.grayscale_histogram)
        self.grayscale_absolute_deviation = np.mean(np.absolute(self.grayscale_histogram - self.grayscale_mean))
        self.grayscale_standard_deviation = np.std(self.grayscale_histogram)

        self.blue_histogram = cv.calcHist([self.image], [0], None, [256], [0, 256])
        self.blue_mean = np.mean(self.blue_histogram)
        self.blue_variance = np.var(self.blue_histogram)
        self.blue_skewness = sp.skew(self.blue_histogram)
        self.blue_kurtosis = sp.kurtosis(self.blue_histogram)
        self.blue_absolute_deviation = np.mean(np.absolute(self.blue_histogram - self.blue_mean))
        self.blue_standard_deviation = np.std(self.blue_histogram)

        self.green_histogram = cv.calcHist([self.image], [1], None, [256], [0, 256])
        self.green_mean = np.mean(self.green_histogram)
        self.green_variance = np.var(self.green_histogram)
        self.green_skewness = sp.skew(self.green_histogram)
        self.green_kurtosis = sp.kurtosis(self.green_histogram)
        self.green_absolute_deviation = np.mean(np.absolute(self.green_histogram - self.green_mean))
        self.green_standard_deviation = np.std(self.green_histogram)

        self.red_histogram = cv.calcHist([self.image], [2], None, [256], [0, 256])
        self.red_mean = np.mean(self.red_histogram)
        self.red_variance = np.var(self.red_histogram)
        self.red_skewness = sp.skew(self.red_histogram)
        self.red_kurtosis = sp.kurtosis(self.red_histogram)
        self.red_absolute_deviation = np.mean(np.absolute(self.red_histogram - self.red_mean))
        self.red_standard_deviation = np.std(self.red_histogram)

        self.histogram_statistics = [self.grayscale_mean, self.grayscale_variance, self.grayscale_skewness[0],
        self.grayscale_kurtosis[0], self.grayscale_absolute_deviation, self.grayscale_standard_deviation,
        self.blue_mean, self.blue_variance, self.blue_skewness[0], self.blue_kurtosis[0], self.blue_absolute_deviation, self.blue_standard_deviation,
        self.green_mean, self.green_variance, self.green_skewness[0], self.green_kurtosis[0], self.green_absolute_deviation, self.green_standard_deviation,
        self.red_mean, self.red_variance, self.red_skewness[0], self.red_kurtosis[0], self.red_absolute_deviation, self.red_standard_deviation]

        self.features.extend(self.histogram_statistics)

    def calculate_HOG(self):
        ''' Calculates the HOG descriptor '''
        hog = cv.HOGDescriptor()
        hog_descriptor = hog.compute(self.grayscale)

        self.features.append(np.var(hog_descriptor))

    def find_dominant_colors(self):
        ''' Calculates the dominant colors '''
        data = np.float32(np.reshape(self.image, (self.height * self.width, 3)))

        _, _, centers = cv.kmeans(data, 2, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv.KMEANS_RANDOM_CENTERS)

        centers = centers.ravel()
        self.features.extend(centers)

    def find_average_color(self):
        ''' Calculates the average color '''
        self.average_color = np.average(np.average(self.image, axis=0), axis=0)

        self.features.extend(self.average_color)

    def find_corners(self):
        ''' Calculates the corners '''
        corners = np.int0(cv.goodFeaturesToTrack(self.grayscale, 2, 0.01, 1))
        corners = corners.ravel()

        self.features.append(np.var(corners))

    def find_edges(self):
        ''' Calculates the edges '''
        edges = cv.Canny(self.grayscale, 10, 126)

        self.features.append(np.var(edges))

    def find_circles(self):
        ''' Calculates the circles '''
        blurred = cv.medianBlur(self.grayscale, 5)
        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, blurred.shape[0]/8,
        param1=100, param2=30, minRadius=1, maxRadius=30)
        
        if circles is not None:
            circles = circles.ravel()
            self.features.append(np.var(circles))
        else:
            self.features.append(0)

    def calculate_thumbnail(self):
        ''' Calculates the thumbnail '''
        self.thumbnail = cv.resize(self.image, (2, 2), interpolation=cv.INTER_AREA)
        self.thumbnail_mean = np.mean(self.thumbnail)
        self.thumbnail_variance = np.var(self.thumbnail)
        self.thumbnail_absolute_deviation = np.mean(np.absolute(self.thumbnail - self.thumbnail_mean))
        self.thumbnail_standard_deviation = np.std(self.thumbnail)

        self.thumbnail_statistics = [self.thumbnail_mean, self.thumbnail_variance,
        self.thumbnail_absolute_deviation, self.thumbnail_standard_deviation]

        self.features.extend(self.thumbnail_statistics)

    def write_features(self):
        ''' Writes the features to a csv file '''
        self.features = pd.DataFrame(self.features).T
        self.features.to_csv('features.csv', index=False, header=False, mode='a')

    def show_image(self, attributes):
        ''' Shows the image table '''
        table = np.hstack(attributes)
        cv.imshow('image', table)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def write_image(self, attributes):
        ''' Writes the image table to a file '''
        split = int(np.ceil(len(attributes)/2))
        table = np.hstack(attributes[:split])
        table = np.vstack([table, np.hstack(attributes[split:])])
        cv.imwrite('sith.jpg', table)

if __name__ == "__main__":

    base_directory = 'images/'
    for _ in range(0, 113):
        test = ImageClassification(base_directory + str(_) + '.jpg')
        test.write_features()
 