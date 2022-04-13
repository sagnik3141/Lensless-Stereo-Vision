import os
import sys
import numpy as np
from PIL import Image
import imageio
from skimage import restoration
import argparse
import glob
from tqdm import tqdm

from RGB2Lensless import RGB2Lensless, loadCropImg, loadPSF

def wienerDeconv(meas, psf, args):
    """
    Returns Deconvolved image given lensless measurement and psf
    """
    return np.stack([restoration.wiener(meas[:,:,i], psf[:,:,i], args.wiener_param) for i in range(3)], axis = 2)

def getArgs():
    """
    Command Line Arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--source_dir',
        type=str,
        help="Path to Source Directory",
        required=True)
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Path to Output Directory",
        required=True)
    parser.add_argument(
        '--psf_path',
        type=str,
        help="Path to PSF",
        default="/home/sagnik/Documents/FlyingThings3D/basler_phlatcam_psf_binned2x_320x448_rgb.npy")
    parser.add_argument(
        '--wiener_param',
        type=int,
        help="Wiener Regularization",
        default=0.1)

    args = parser.parse_args()

    return args


def main():
    
    args = getArgs()

    ### Get File List ###
    img_list = os.listdir(args.source_dir)
    psf = loadPSF(args.psf_path)
    save_num = 0

    ### Create Subdirectories ###
    lensless_dir = os.path.join(args.output_dir, "lensless")
    if not os.path.exists(lensless_dir):
        os.makedirs(lensless_dir)
    GT_dir = os.path.join(args.output_dir, "GT")
    if not os.path.exists(GT_dir):
        os.makedirs(GT_dir)
    input_dir = os.path.join(args.output_dir, "input")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    ### Create Dataset ###
    for img_name in tqdm(img_list):
        img = loadCropImg(os.path.join(args.source_dir, img_name))
        meas = RGB2Lensless(img, psf)
        imageio.imwrite(os.path.join(lensless_dir, f"{save_num}.png"), (meas * 255).astype(np.uint8))
        deconvolved = wienerDeconv(meas, psf, args)
        imageio.imwrite(os.path.join(input_dir, f"{save_num}.png"), (deconvolved * 255).astype(np.uint8))
        save_num+=1





        



if __name__=="__main__":
    main()