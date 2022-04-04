import numpy as np
from PIL import Image
from scipy import fft


def channel2Lensless(imgC, psfC):
    """
    Convolution of imgC and psfC in the frequency domain.

    Arguments:
    imgC - Single Channel of image
    psfC - Single Channel of psf

    Returns:
    lenslessC - Lensless Measurement (Single Channel)
    """

    shape = [
        imgC.shape[i] +
        psfC.shape[i] -
        1 for i in range(
            imgC.ndim)]  # Target Shape
    imgC_dft = fft.fftn(imgC, shape)  # DFT of Image
    psfC_dft = fft.fftn(psfC, shape)  # DFT of PSF
    # Convolution is Multiplication in Frequency Domain
    lenslessC = fft.ifftn(imgC_dft * psfC_dft, shape)

    return lenslessC


def RGB2Lensless(img, psf):
    """
    Convolution of img (RGB) and psf in the frequency domain.

    Arguments:
    img - RGB Input Image to be convolved (channel dim last)
    psf - Convolution Kernel

    Returns:
    lenslessRGB - Lensless Measurement
    """

    lenslessRGB = [channel2Lensless(
        img[:, :, i], psf[:, :, i]) for i in range(3)]
    lenslessRGB = np.stack(lenslessRGB, axis=2)
    lenslessRGB = np.abs(lenslessRGB)

    return lenslessRGB


def getArgs():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
