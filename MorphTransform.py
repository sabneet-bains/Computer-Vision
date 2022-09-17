import numpy as np
import cv2 as cv

class MorphTransform():
    ''' Morphological Transformations '''
    def __init__(self, image_path):
        self.image = cv.imread(image_path, 0)
        self.copy = self.image.copy()
        self.eroded = None
        self.dilated = None
        self.opening = None
        self.closing = None
        self.gradient = None
        self.tophat = None
        self.blackhat = None
        self.hitmiss = None

# methods
    def apply_morphology(self, morph_type, kernel_size):
        ''' Performs the specified type of morphological operation on the image '''
        if morph_type == 'erosion':
            self.eroded = cv.erode(self.copy, kernel_size, iterations=1)
            return self.eroded
        elif morph_type == 'dilation':
            self.dilated = cv.dilate(self.copy, kernel_size, iterations=1)
            return self.dilated
        elif morph_type == 'opening':
            self.opening = cv.morphologyEx(self.copy, cv.MORPH_OPEN, kernel_size)
            return self.opening
        elif morph_type == 'closing':
            self.closing = cv.morphologyEx(self.copy, cv.MORPH_CLOSE, kernel_size)
            return self.closing
        elif morph_type == 'gradient':
            self.gradient = cv.morphologyEx(self.copy, cv.MORPH_GRADIENT, kernel_size)
            return self.gradient
        elif morph_type == 'tophat':
            self.tophat = cv.morphologyEx(self.copy, cv.MORPH_TOPHAT, kernel_size)
            return self.tophat
        elif morph_type == 'blackhat':
            self.blackhat = cv.morphologyEx(self.copy, cv.MORPH_BLACKHAT, kernel_size)
            return self.blackhat
        elif morph_type == 'hitmiss':
            self.hitmiss = cv.morphologyEx(self.copy, cv.MORPH_HITMISS, kernel_size)
            return self.hitmiss
        else:
            print('Invalid Morph Type')
            return None

    def write_image(self, attributes):
        ''' Writes the image table to a file '''
        split = int(np.ceil(len(attributes)/2))
        table = np.hstack(attributes[:split])
        table = np.vstack([table, np.hstack(attributes[split:])])
        cv.imwrite('morphology.png', table)

    def show_image(self, attributes):
        ''' Shows the image table '''
        split = int(np.ceil(len(attributes)/2))
        table = np.hstack(attributes[:split])
        table = np.vstack([table, np.hstack(attributes[split:])])
        cv.imshow('image', table)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    avatar = MorphTransform('avatar.png')

    dilateKernel = np.ones((5,5),np.uint8)
    imgMask2 = cv2.dilate(imgMask,dilateKernel,iterations=5)
    imgMask2 = cv2.erode(imgMask2,dilateKernel,iterations=5)
    imgMask2 = cv2.dilate(imgMask2,dilateKernel,iterations=5)
    imgMask2 = cv2.erode(imgMask2,dilateKernel,iterations=5)

    # avatar.apply_morphology('erosion', np.ones((5,5), np.uint8))
    # avatar.apply_morphology('dilation', np.ones((5,5), np.uint8))
    # avatar.apply_morphology('opening', np.ones((5,5), np.uint8))
    # avatar.apply_morphology('closing', np.ones((5,5), np.uint8))
    # avatar.apply_morphology('gradient', np.ones((5,5), np.uint8))
    # avatar.apply_morphology('tophat', np.ones((5,5), np.uint8))
    # avatar.apply_morphology('blackhat', np.ones((5,5), np.uint8))
    # avatar.apply_morphology('hitmiss', np.ones((5,5), np.uint8))

    # avatar.show_image([avatar.eroded, avatar.dilated,
    # avatar.opening, avatar.closing, avatar.gradient, avatar.tophat,
    # avatar.blackhat, avatar.hitmiss])

    # avatar.write_image([avatar.eroded, avatar.dilated,
    # avatar.opening, avatar.closing, avatar.gradient, avatar.tophat,
    # avatar.blackhat, avatar.hitmiss])
