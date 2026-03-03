"""Project ML Algorithm-Blake Kerr"""

import os
import datetime

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2  # for image resizing

from utils.utils import (
    init_wandb,
    log_to_wandb,
    log_image_to_wandb,
    finish_wandb,
    plot_cm,
    set_seed,
)

USE_WANDB = False  # Set to True to enable Weights & Biases logging
USE_SEED = False  # Set to True to enable reproducibility
RANDOM_SEED = 42  # Seed value for reproducibility
TRAIN_FLAG = True  # Set to True to force retraining the model
USE_TEST = False  # Set to True to visualize test set instead of train set

# Training configuration
EPOCHS = 20  # Number of training epochs
LR = 0.0003  # Learning rate
BATCH_SIZE = 8  # Batch size

# File paths
fig_path = "./figures"
model_path = "./models"
run_timestamp = datetime.datetime.now().strftime("%m%d-%H%M")

# WANDB configuration
WANDB_PROJECT = "lab4-cat-classifier"
WANDB_RUN_NAME = f"lab4sol-{run_timestamp}"
data_dir = "/remote_home/Project_Data"
os.makedirs(data_dir, exist_ok=True)

def get_dataloaders(data_dir, batch_size=BATCH_SIZE, target_size=(256,256)):

    class NPYKeypointDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

        def __len__(self):
            return len(self.files)

        def parse_point_from_filename(self, filename):
            base = os.path.basename(filename)
            base = os.path.splitext(base)[0]
            parts = base.split("_")
            x_act = int(parts[1])
            y_act = int(parts[3])
            intensity = int(parts[5])
            return [x_act, y_act]

        def __getitem__(self, idx):
            file = self.files[idx]
            path = os.path.join(self.data_dir, file)

            image = np.load(path)

            # Convert to float32 before resizing (needed for cv2.resize)
            image = image.astype(np.float32)

            # Resize image to target_size (H, W)
            image = cv2.resize(image, target_size)

            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image = image / 255.0

            x, y = self.parse_point_from_filename(file)

            # Normalize coordinates to resized image
            # Normalize coordinates relative to resized image
            # Normalize coordinates relative to original image (1024x1024)
            depth, h ,w= image.shape
            target = torch.tensor([x / w, y / h], dtype=torch.float32)

            return image, target

    full_dataset = NPYKeypointDataset(data_dir)

    # 70 / 15 / 15 split
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Add num_workers and pin_memory for faster GPU transfer
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, valloader, testloader

class SingleObjectDetector(nn.Module):
    def __init__(self):
        super(SingleObjectDetector, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 512x512
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # → 256x256

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # → 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # → 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)               # → 32x32
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 32 * 32, 128),  # 131072 → 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Output layer (coords only — presence removed)
        self.coords = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv_layers(x)
        # Flatten full spatial map
        x = torch.flatten(x, 1)  # shape: [batch, 131072]
        x = self.fc(x)
        coords = self.coords(x)
        return coords

def evaluate_model(model, dataloader, device, criterion, original_size=(1024, 1024)):
    """
    Evaluate a model on a given dataloader for keypoint regression.

    Args:
        model: PyTorch model
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on (cpu/cuda)
        criterion: Loss function (optional, computes loss if provided)
        original_size: tuple (height, width) of the original images

    Returns:
        dict with keys:
            - 'predictions': numpy array of predicted coordinates (normalized)
            - 'true_labels': numpy array of true coordinates (normalized)
            - 'loss': float, average loss (only if criterion provided)
            - 'accuracy': float, % of points within threshold pixels
    """
    model.eval()
    y_pred = []
    y_true = []
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            coords = model(images)

            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(coords, targets)
                total_loss += loss.item() * images.size(0)

            y_pred.extend(coords.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    orig_h, orig_w = original_size

    # Scale normalized coordinates to original pixels
    y_pred_pixels = y_pred.copy()
    y_pred_pixels[:, 0] *= orig_w
    y_pred_pixels[:, 1] *= orig_h

    y_true_pixels = y_true.copy()
    y_true_pixels[:, 0] *= orig_w
    y_true_pixels[:, 1] *= orig_h
    # Compute distances in pixels
    threshold_pixels = 10
    distances = np.linalg.norm(y_pred_pixels - y_true_pixels, axis=1)
    correct_count = np.sum(distances <= threshold_pixels)
    accuracy = correct_count / len(distances) * 100

    results = {
        'predictions': y_pred_pixels,   # scaled to original image
        'true_labels': y_true_pixels,   # scaled to original image
        'accuracy': accuracy,
    }

    if criterion is not None:
        results['loss'] = total_loss / len(dataloader.dataset)

    return results

def train_model(model, trainloader, valloader, criterion, optimizer, device, epochs, use_wandb=False):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in trainloader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            coords = model(images)
            loss = criterion(coords, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        val_results = evaluate_model(model, valloader, device, criterion)
        val_loss = val_results['loss']
        val_accuracy = val_results['accuracy']

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if use_wandb:
            log_to_wandb({"train_loss": train_loss, "val_loss": val_loss}, step=epoch, use_wandb=use_wandb)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    print("Finished Training")
    return model, history

def main():
    if USE_SEED:
        set_seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    if "cuda" in str(device):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("*" * 60)
        print("WARNING Using CPU (this will be slow!)")
        print("*" * 60)

    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    trainloader, valloader, testloader = get_dataloaders(data_dir, batch_size=BATCH_SIZE, target_size=(512, 512))

    config = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "seed": RANDOM_SEED if USE_SEED else None}
    init_wandb(project_name=WANDB_PROJECT, run_name=WANDB_RUN_NAME, config=config, use_wandb=USE_WANDB)

    model_save_path = f"{model_path}/single_object_detector.pth"
    model = SingleObjectDetector().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    if not os.path.exists(model_save_path) or TRAIN_FLAG:
        model, history = train_model(model, trainloader, valloader, criterion, optimizer, device, epochs=EPOCHS, use_wandb=USE_WANDB)
        torch.save(model.state_dict(), model_save_path)
    else:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

    loader = testloader if USE_TEST else trainloader
    results = evaluate_model(model, loader, device, criterion)
    y_pred_pixels = results['predictions']   # Already scaled to 1024x1024
    y_true_pixels = results['true_labels']   # Already scaled to 1024x1024
    accuracy = results['accuracy']

    # Print a few predictions vs true values
    print("\nSample predictions vs true coordinates:")
    for i in range(min(5, len(y_pred_pixels))):
        print(f"Predicted: {y_pred_pixels[i]}, True: {y_true_pixels[i]}")

    errors = np.linalg.norm(y_pred_pixels - y_true_pixels, axis=1)
    print(f"\nMean pixel error: {np.mean(errors):.2f}, Max error: {np.max(errors):.2f}, Accuracy: {results['accuracy']:.4f}")

    finish_wandb(use_wandb=USE_WANDB)


if __name__ == "__main__":
    main()