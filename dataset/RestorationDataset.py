import os
import random
import numpy as np
from PIL import Image
import torch
from scipy import fft
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    
    def __init__(self, args, psf):
        
        self.source_dir = args.source_dir
        self.img_list = os.listdir(args.source_dir)
        self.psf = psf

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img = self.loadCropImg(os.path.join(self.source_dir, self.img_list[index]))
        meas = self.RGB2Lensless(img, self.psf)
        meas = np.transpose(meas, (2, 0, 1))
        meas = torch.from_numpy(meas)
        img = np.transpose(img, (2, 0, 1))

        return img, meas

        

    def loadCropImg(self, img_path):
        """
        Loads and Crops Image
        """
        img = np.array(Image.open(img_path))
        img = img / 255.0
        img = self.cropImg(img)

        return img


    def cropImg(self, img, size=(320, 320)):
        """
        Returns Random Cropped image given target size.
        """
        dim1, dim2, _ = img.shape
        c1 = np.random.choice(dim1 - size[0])
        c2 = np.random.choice(dim2 - size[1])

        return img[c1:c1 + 320, c2:c2 + 320]

    def channel2Lensless(self, imgC, psfC):
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


    def RGB2Lensless(self, img, psf):
        """
        Convolution of img (RGB) and psf in the frequency domain.

        Arguments:
        img - RGB Input Image to be convolved (channel dim last)
        psf - Convolution Kernel

        Returns:
        lenslessRGB - Lensless Measurement
        """

        lenslessRGB = [self.channel2Lensless(
            img[:, :, i], psf[:, :, i]) for i in range(3)]
        lenslessRGB = np.stack(lenslessRGB, axis=2)
        lenslessRGB = np.abs(lenslessRGB)
        noise = np.random.normal(0, 0.01, (lenslessRGB.shape[0], lenslessRGB.shape[1])) # Add 40dB noise
        lenslessRGB = lenslessRGB+noise[:,:,np.newaxis]

        return lenslessRGB

