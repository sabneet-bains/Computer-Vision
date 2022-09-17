import cv2 as cv
import numpy as np

class CoinDetection():

    # constructor
    def __init__(self, image_path):
        ''' Initializes the class '''
        self.image = cv.imread(image_path)
        self.copy = self.image.copy()
        self.grayscaled = []
        self.blurred = []
        self.edged = []
        self.dilated = []
        self.contours = []
        self.cleaned = []
        self.thresholded = []
        self.masked = self.copy
        self.labeled = self.copy

    # getters
    def get_image(self):
        ''' Returns the image '''
        return self.image

    # methods
    def convert_to_grayscale(self):
        ''' Converts the image to grayscale '''
        self.grayscaled = cv.cvtColor(self.copy, cv.COLOR_BGR2GRAY)

        return self.grayscaled

    def apply_blur(self, blur_type, kernel_size):
        ''' Applies a blur (smoothing) to the image '''
        if blur_type == 'gaussian':
            self.blurred = cv.GaussianBlur(self.grayscaled, (kernel_size, kernel_size), 0)
        
        elif blur_type == 'median':
            self.blurred = cv.medianBlur(self.grayscaled, kernel_size)
        
        elif blur_type == 'bilateral':
            self.blurred = cv.bilateralFilter(self.grayscaled, kernel_size, 75, 75)
        
        elif blur_type == 'box':
            self.blurred = cv.boxFilter(self.grayscaled, -1, (kernel_size, kernel_size))
        
        elif blur_type == 'mean':
            self.blurred = cv.blur(self.grayscaled, (kernel_size, kernel_size))
        
        return self.blurred

    def remove_background(self):
        ''' Removes the background from the image '''
        baseline = cv.threshold(self.blurred,127,255, cv.THRESH_TRUNC)[1]
        background = cv.threshold(baseline,100,255, cv.THRESH_BINARY)[1]
        foreground = cv.threshold(baseline,100,255, cv.THRESH_BINARY_INV)[1]

        foreground = cv.bitwise_and(self.copy, self.copy, mask=foreground)
        background = cv.cvtColor(background, cv.COLOR_GRAY2BGR)
        self.cleaned = background + foreground

        return self.cleaned

    def apply_otsu_threshold(self, threshold):
        ''' Applies an OTSU threshold to the grayscale image '''
        self.thresholded = cv.threshold(self.blurred, threshold, 255, cv.THRESH_OTSU)[1]
        self.thresholded = cv.bitwise_not(self.thresholded) # Pseudo mask

        return self.thresholded

    def apply_mask(self):
        ''' Applies a binary mask to the image '''
        self.convert_to_grayscale()
        self.apply_blur('gaussian', 9)
        self.apply_otsu_threshold(threshold=127)
        self.masked = cv.bitwise_and(self.copy, self.copy, mask=self.thresholded)

        return self.masked
        
    def apply_edge_detection(self, detection_type, threshold1, threshold2):
        ''' Applies canny edge detection to the image '''
        if detection_type == 'canny':
            self.edged = cv.Canny(self.blurred, threshold1, threshold2, apertureSize=3)
        
        elif detection_type == 'sobel':
            self.edged = cv.Sobel(self.blurred, -1, 1, 1)
        
        elif detection_type == 'laplacian':
            self.edged = cv.Laplacian(self.blurred, -1)
        
        elif detection_type == 'scharr':
            self.edged = cv.Scharr(self.blurred, -1, 1, 0)
        
        return self.edged
    
    def apply_dilation(self, kernel_size):
        ''' Applies a dilation to the image '''
        self.dilated = cv.dilate(self.edged, (kernel_size, kernel_size), iterations = 2)

        return self.dilated

    def find_contours(self):
        ''' Finds the contours in the image '''
        self.contours = cv.findContours(self.dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        print('There are', len(self.contours), 'objects in the image') # Count blobs
        
        return self.contours

    def label_coins(self):
        ''' Labels the coins in the image '''
        self.convert_to_grayscale()
        self.apply_blur('gaussian', 9)
        self.apply_edge_detection('canny', 30, 150)
        self.apply_dilation(1)
        self.find_contours()

        for c in self.contours:
            M = cv.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))

            length = cv.arcLength(c, True)
            # area = cv.contourArea(c)

            # dime < penny < nickel < quarter
            if length > 382 and length < 390:
                cv.drawContours(self.labeled, [c], -1, (255, 255, 0), 2)
                cv.putText(self.labeled, "Dime", (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
     
            if length > 390 and length < 398:
                cv.drawContours(self.labeled, [c], -1, (0, 0, 255), 2)
                cv.putText(self.labeled, "Penny", (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if length > 440 and length < 448:
                cv.drawContours(self.labeled, [c], -1, (255, 0, 0), 2)
                cv.putText(self.labeled, "Nickel", (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if length > 495 and length < 514:
                cv.drawContours(self.labeled, [c], -1, (0, 255, 0), 2)
                cv.putText(self.labeled, "Quarter", (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
   
        return self.labeled

    def write_image(self, attributes):
        ''' Writes the image to a file '''
        table1 = np.hstack(attributes)
        # table2 = np.vstack([cv.imread('result.jpg'), table1])
        cv.imwrite('result.jpg', table1)

    def show_image(self, attributes):
        ''' Shows the image '''
        table = np.hstack(attributes)
        cv.imshow('image', table)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    coins = CoinDetection('image_00.jpg')
    coins.apply_mask()
    coins.label_coins()
    coins.write_image([coins.image, coins.masked, coins.labeled])
