import numpy as np
import cv2 as cv

class DiscreteFourierTransform():
    def __init__(self, image_path):
        self.image = cv.imread(image_path, 0)

    # methods
    def apply_dft_transform(self):
        ''' Performs the discrete fourier transform on the image '''
        self.dft = cv.dft(np.float32(self.image), flags=cv.DFT_COMPLEX_OUTPUT)
        self.dft_shift = np.fft.fftshift(self.dft)
        self.magnitude, self.phase = cv.cartToPolar(self.dft_shift[:,:,0], self.dft_shift[:,:,1])
        self.real, self.imaginary = cv.polarToCart(self.magnitude, self.phase)
        self.combined = cv.merge([self.real, self.imaginary])

    def apply_idft_transform(self):
        ''' Performs the inverse discrete fourier transform on the image '''
        self.idft_shift = np.fft.ifftshift(self.combined)
        self.idft = cv.idft(self.idft_shift)
        self.idft = cv.magnitude(self.idft[:,:,0], self.idft[:,:,1])
        self.identity = cv.normalize(self.idft, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


if __name__ == "__main__":
    surreal = DiscreteFourierTransform('surreal.jpg')
    surreal.apply_dft_transform()
    surreal.apply_idft_transform()

    cv.imwrite('Original.jpg', surreal.image)
    cv.imwrite('Magnitude.jpg', surreal.real)
    cv.imwrite('Identity.jpg', surreal.identity)
    cv.waitKey(0)
    cv.destroyAllWindows()
