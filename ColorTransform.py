import numpy as np
import cv2 as cv

class ColorTransform():
    ''' Color Model Transformations '''
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.copy = self.image.copy()
        self.hsv = None
        self.hue = None
        self.saturation = None
        self.value = None

    def extract_hsv(self):
        ''' Extracts the HSV channels '''
        self.hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        self.hue, self.saturation, self.value = cv.split(self.hsv)
        return self.hue, self.saturation, self.value

    def shift_hue(self, hue):
        ''' Shifts the hue of the image '''
        self.extract_hsv()
        self.hue = cv.add(self.hue, hue)
        self.hsv = cv.merge((self.hue, self.saturation, self.value))
        self.copy = cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR)
        return self.copy

    def show_image(self, attributes):
        ''' Shows the image table '''
        split = int(np.ceil(len(attributes)/2))
        table = np.hstack(attributes[:split])
        table = np.vstack([table, np.hstack(attributes[split:])])
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
    grogu = ColorTransform('grogu.jpg')
    grogu.shift_hue(hue=110)
    grogu.show_image([grogu.image, grogu.copy])
    grogu.write_image([grogu.image, grogu.copy])
