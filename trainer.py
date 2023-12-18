"""Trainer module"""
import numpy as np

from typing import Dict, Any

import torch

import matplotlib.pyplot as plt

from tqdm import tqdm


def train(**inputs: Dict[str, Any]) -> None:
    """trains the model given the inputs

    Parameters:
    -----------
        model: torch.nn.Module
            the model to train
        criterion: torch.nn.Loss
            the loss function
        optimizer: torch.optim.Optimizer
            the optimizer
        trainloader: torch.utils.data.DataLoader
            the training data loader
        validloader: torch.utils.data.DataLoader
            the validation data loader
        epochs: int
            number of epochs to train
        device: str
            device to train on
        scheduler: torch.optim.lr_scheduler
            the learning rate scheduler
        training_losses: List[float]
            list to store training losses
        validation_losses: List[float]
            list to store validation losses
    """
    model = inputs["model"]

    if model is None:
        raise ValueError("model cannot be None")

    criterion = inputs["criterion"]
    optimizer = inputs["optimizer"]
    trainloader = inputs["trainloader"]
    validloader = inputs["validloader"]
    epochs = inputs["epochs"]
    device = inputs["device"]
    scheduler = inputs.get("scheduler")
    training_losses = inputs["training_losses"]
    validation_losses = inputs["validation_losses"]
    verbose = inputs.get("verbose", True)

    print("Training on {}...".format(device))

    # load pretrained model if available
    if inputs.get("pretrained_model"):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(inputs["pretrained_model"]))

    # move model to device
    model.to(device)

    # set bestloss to track the best validation loss and save the best model
    best_loss = np.inf

    # loop through epochs
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0

        # set model to train mode and train
        model.train()
        # loop through batches
        for left, right, ldepth, rdepth in tqdm(trainloader, desc="Training"):
            # concatenate left and right images
            diff = torch.abs(left - right).mean(dim=1, keepdim=True)
            x = torch.cat((left, diff), dim=1)
            x = x.to(device)
            ldepth = ldepth.to(device)

            pred_depth = model(x)
            loss = criterion(pred_depth, ldepth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # validate
        model.eval()
        with torch.no_grad():
            for left, right, ldepth, rdepth in tqdm(validloader, desc="Validating"):
                # concatenate left and right images
                diff = torch.abs(left - right).mean(dim=1, keepdim=True)
                x = torch.cat((left, diff), dim=1)
                x = x.to(device)
                ldepth = ldepth.to(device)

                pred_depth = model(x)
                loss = criterion(pred_depth, ldepth)

                valid_loss += loss.item()

            # for visualization purposes
            left, right, ldepth, _ = next(iter(validloader))
            diff = torch.abs(left - right).mean(dim=1, keepdim=True)
            x = torch.cat((left, diff), dim=1)
            x = x.to(device)
            pred_depth = model(x)

        # verbose
        training_losses[epoch] = train_loss / len(trainloader)
        validation_losses[epoch] = valid_loss / len(validloader)

        print(
            f"Epoch: {epoch+1}/{epochs}.. Training Loss: {training_losses[epoch]:.6f}.. Validation Loss: {validation_losses[epoch]:.6f}"
        )

        if validation_losses[epoch] < best_loss:
            print(
                "loss decreased from {:.6f} to {:.6f} -> saving best model.".format(
                    best_loss, validation_losses[epoch]
                )
            )
            best_loss = validation_losses[epoch]
            torch.save(model.state_dict(), "best-model.pth")

        if scheduler:
            # scheduler.step(valid_loss / len(validloader))
            scheduler.step()

        if verbose:
            # display right left and depth images
            fig, ax = plt.subplots(1, 4, figsize=(15, 15))
            ax[0].imshow(left[0].permute(1, 2, 0).detach().cpu().numpy())
            ax[0].set_title("Left Image")
            ax[1].imshow(right[0].permute(1, 2, 0).detach().cpu().numpy())
            ax[1].set_title("Right Image")
            ax[2].imshow(pred_depth[0].detach().cpu().squeeze().numpy())
            ax[2].set_title("Depth Image")
            ax[3].imshow(ldepth[0].detach().cpu().squeeze().numpy())
            ax[3].set_title("Ground Truth Depth Image")
            plt.show()

            plt.plot(training_losses, label="Training loss")
            plt.plot(validation_losses, label="Validation loss")
            plt.legend(frameon=False)
            plt.show()
