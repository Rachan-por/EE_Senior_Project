"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import forward
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchvision.utils import save_image
import wandb
import utils

def train_step(model: torch.nn.Module,
               T: int,
               df,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float]:

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X = X.to(device)

        # Random timestep
        t = torch.randint(0, T, (1,), device=device).long()

        # 1. Forward pass
        x_noisy, noise = df.forward_diffusion_sample(X, t, device)
        noise_pred = model(x_noisy, t)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(noise, noise_pred)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss per batch
    train_loss = train_loss / len(dataloader)

    return train_loss


def train(model: torch.nn.Module,
          T: int,
          df,
          train_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    # Create empty results dictionary
    results = {"train_loss": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                T=T,
                                df=df,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)

        # Print out what's happening
        print(
            f"\n"
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} |\n ------------- "
        )
        
        # Save model with the lowest loss
        if results["train_loss"][-1] >= train_loss:
            utils.save_model(model=model,
                     target_dir="models",
                     model_name="best_diffusion.pth")

        # sampling
        imgs = df.sample_image(model, T)
        img = imgs[-1]

        # log
        wandb.log({"loss": train_loss, "Sampling Image": wandb.Image(img)})
        
        # Update results dictionary
        results["train_loss"].append(train_loss)

    # Return the filled results at the end of the epochs
    return results
