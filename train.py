"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse
import torch
import data_setup
import engine
import model_builder
import utils
import loss_function
import forward
from torchvision import transforms
import wandb


# Setup hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str)
parser.add_argument("--timesteps", type=int)
parser.add_argument("--loss", type=str, help="l1 or l2")
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)

# python train.py --train_dir /Users/rachan/Desktop/small_train --timesteps 50 --loss "l1" --num_epochs 3 --batch_size 1 --lr 0.001
args = parser.parse_args()

train_dir = args.train_dir
timesteps = args.timesteps
loss_type = args.loss
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

df = forward.Diffusion(timesteps, device=device)

# Create transforms
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader = data_setup.create_dataloaders(
    train_dir=train_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.SimpleUnet().to(device)

# Set loss and optimizer
loss_fn = loss_function.loss(loss_type)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

if __name__ == '__main__':
    wandb.login(key='f110a602e329c7d1f51a3e816bdcea9d1a547243')

    wandb.init(
        # Set the project where this run will be logged
        project="Test_Sampling"
    )

    # Start training with help from engine.py
    engine.train(model=model,
                 df=df,
                 T=timesteps,
                 train_dataloader=train_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="last_diffusion.pth")

    wandb.finish()
