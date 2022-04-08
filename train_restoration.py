import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from models.unet_128 import Unet
from utils.model_serialization import load_state_dict
from dataset.dataloader import create_dataloaders


def train(model, train_loader, val_loader, args, device):
    
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.L1Loss()

    ### Training ###

    writer = SummaryWriter('runs/train') # Tensorboard for plots
    best_val_loss = None
    counter = 0
    iterator = tqdm(range(args.num_epochs))
    for i in iterator:

        ### Weight Updates ###
        
        epoch_loss = 0
        for b, (X_train, Y_train) in enumerate(train_loader):
            X_train = X_train.to(device).float()
            Y_train = Y_train.to(device).float()
            Y_pred = model(X_train)

            loss = criterion(Y_pred, Y_train)
            epoch_loss+=loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss/b

        ### Validation ###
        
        val_loss = 0
        for b, (X_val, Y_val) in enumerate(val_loader):
            X_val = X_val.to(device).float()
            Y_val = Y_val.to(device).float()
            Y_pred = model(X_val)

            loss = criterion(Y_pred, Y_val)
            val_loss+=loss
            ### TODO: Add few images to tensorboard###

        val_loss = val_loss/b
        iterator.set_postfix({'train_loss': epoch_loss.item(), 'val_loss': val_loss.item()})

        # Early Stopping
        if best_val_loss is None:
            best_val_loss = val_loss
        elif val_loss>best_val_loss:
            counter+=1
            if counter>args.patience:
                iterator.close()
                print(f"Early stopping at epoch {i+1}.")
                break
        else:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'best_weights.pt'))

        # Plotting
        writer.add_scalars('Loss', {
            'Train': epoch_loss,
            'Validation': val_loss,
        }, i+1)


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
    train(model, train_loader, val_loader, args, device)


if __name__ == "__main__":
    main()
