import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter

from models.unet_128 import Unet
from utils.model_serialization import load_state_dict
from dataset.dataloader import create_dataloaders
from dataset.RestorationDataset import ImageDataset
from wiener_model import wienerModel
from RGB2Lensless import loadPSF
from utils.ops import unpixel_shuffle, rggb_2_rgb, rgb_2_rggb
from utils.vgg_loss import vgg_loss

def train(model, wiener_model, psf, train_loader, val_loader, args, device):

    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = vgg_loss

    # PSF to Torch
    psf = np.transpose(psf, (2, 0, 1))[np.newaxis,:,:,:]
    psf = torch.from_numpy(psf)
    psf = psf.to(device).float()

    ### Training ###

    writer = SummaryWriter('runs/train')  # Tensorboard for plots
    best_val_loss = None
    counter = 0
    iterator = tqdm(range(args.num_epochs))
    for i in iterator:

        ### Weight Updates ###

        epoch_loss = 0
        for b, (img, meas) in enumerate(train_loader):
            #X_train = X_train.to(device).float()
            #Y_train = Y_train.to(device).float()
            #Y_pred = model(X_train)

            img = img.to(device).float()
            meas = meas.to(device).float()

            intermediate = wiener_model(meas, psf)
            intermediate_rggb = rgb_2_rggb(intermediate)
            intermediate_unpixel_shuffled = unpixel_shuffle(intermediate_rggb, args.pixelshuffle_ratio)
            output_unpixel_shuffled = model(intermediate_unpixel_shuffled)
            output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)

            loss = criterion(output, img, device)
            epoch_loss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / b

        ### Validation ###

        val_loss = 0
        for b, (img, meas) in enumerate(val_loader):
            img = img.to(device).float()
            meas = meas.to(device).float()

            intermediate = wiener_model(meas, psf)
            intermediate_rggb = rgb_2_rggb(intermediate)
            intermediate_unpixel_shuffled = unpixel_shuffle(intermediate_rggb, args.pixelshuffle_ratio)
            output_unpixel_shuffled = model(intermediate_unpixel_shuffled)
            output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)

            loss = criterion(output, img, device)
            val_loss += loss
            ### TODO: Add few images to tensorboard###

        val_loss = val_loss / b
        iterator.set_postfix(
            {'train_loss': epoch_loss.item(), 'val_loss': val_loss.item()})

        # Early Stopping
        if best_val_loss is None:
            best_val_loss = val_loss
        elif val_loss > best_val_loss:
            counter += 1
            if counter > args.patience:
                iterator.close()
                print(f"Early stopping at epoch {i+1}.")
                break
        else:
            best_val_loss = val_loss
            counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.ckpt_dir,
                    f'best_weights.pt'))

        # Plotting
        writer.add_scalars('Loss', {
            'Train': epoch_loss,
            'Validation': val_loss,
        }, i + 1)


def get_init_unet(args, pretrained):
    """
    Returns UNet128, pretrained if pretrained is True
    """
    model = Unet(args)

    if pretrained:
        ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"))
        load_state_dict(model, ckpt["state_dict"])

    return model


def getArgs():
    
    parser = argparse.ArgumentParser()

    # Unet Args
    parser.add_argument('--pixelshuffle_ratio', type = int, default = 2)
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--num_groups', type = int, default = 8)
    parser.add_argument('--device', type = str, default = "cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--ckpt_path', type = str, required=True)

    parser.add_argument('--reg_param_init', type = int, default = 0.1)

    parser.add_argument('--psf_path', type = str, required = True)

    # Data Args
    parser.add_argument('--source_dir', type = str, required = True)
    parser.add_argument('--val_split', type = float, default = 0.1)
    parser.add_argument('--test_split', type = float, default = 0.1)
    parser.add_argument('--shuffle_data', type = bool, default = True)

    # Train Args
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--patience', type = int, default = 5)
    parser.add_argument('--ckpt_dir', type = str, default = "checkpoints")

    args = parser.parse_args()






def main():

    args = getArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_init_unet(args, pretrained=True)
    model = model.to(device)

    wiener_model = wienerModel(args)
    wiener_model = wiener_model.to(device)

    psf = loadPSF(args.psf_path)
    train_loader, val_loader, _ = create_dataloaders(args, ImageDataset(args, psf))
    train(model, wiener_model, psf, train_loader, val_loader, args, device)


if __name__ == "__main__":
    main()
