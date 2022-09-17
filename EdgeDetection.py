import cv2 as cv

class EdgeDetection():

    # constructor
    def __init__(self, image_path):
        ''' Initializes the class '''
        self.image = cv.imread(image_path)

    # methods
    def convert_to_grayscale(self):
        ''' Converts the image to grayscale '''
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return self.image

    def apply_gaussian_blur(self, kernel_size):
        ''' Applies a gaussian blur to the image '''
        self.image = cv.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        return self.image

    def apply_canny_edge_detection(self, threshold1, threshold2):
        ''' Applies canny edge detection to the image '''
        self.image = cv.Canny(self.image, threshold1, threshold2)
        return self.image

    def write_image(self):
        ''' Writes the image to a file '''
        cv.imwrite('Dracula2.jpg', self.image)

    def show_image(self):
        ''' Shows the image '''
        cv.imshow('image', self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    goku = EdgeDetection('Dracula.jpg')
    goku.convert_to_grayscale()
    goku.apply_gaussian_blur(5)
    goku.apply_canny_edge_detection(100, 200)
    # goku.show_image()
    goku.write_image()
