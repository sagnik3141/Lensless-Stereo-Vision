import numpy as np
from PIL import Image
from scipy import fft
import imageio


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


def loadCropImg(img_path):
    """
    Loads and Crops Image
    """
    img = np.array(Image.open(img_path))
    img = img / 255.0
    img = cropImg(img)

    return img


def cropImg(img):
    pass


def loadPSF(psf_path):
    """
    Loads and normalizes PSF
    """
    psf = np.load(psf_path)
    psf = np.stack([psf[:, :, i] / np.sum(psf[:, :, i])
                   for i in range(3)], axis=2)  # Normalize each channel to 1

    return psf


def getArgs():
    pass


def main():

    args = getArgs()

    if not args.is_dir:
        img = loadCropImg(args.img_path)
        psf = loadPSF(args.psf_path)

        lenslessRGB = RGB2Lensless(img, psf)
        imageio.imwrite(args.save_path, lenslessRGB)

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
