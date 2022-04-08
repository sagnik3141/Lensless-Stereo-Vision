import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from models.unet_128 import Unet
from utils.model_serialization import load_state_dict
from dataset.dataset_wrapper import create_dataloaders

def train():
    pass

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
    pass

def main():
    
    args = getArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_init_unet(args, pretrained=True)
    model = model.to(device)

    train_loader, val_loader, _ = create_dataloaders(args)





if __name__=="__main__":
    main()