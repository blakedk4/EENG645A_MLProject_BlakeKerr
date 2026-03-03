'''1. Architecture idea

Input → Conv layers → Flatten → Fully Connected (FC) layers

Two outputs:

presence: 1 neuron with sigmoid (0–1 probability)

coords: 2 neurons (x, y normalized between 0–1)

We’ll assume images are 64x64 for simplicity.'''

"""Lab 4? TEMPLATE"""

import os
import datetime

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.utils import (
    init_wandb,
    log_to_wandb,
    log_image_to_wandb,
    finish_wandb,
    plot_cm,
    set_seed,
)

########################################################################################
# Configuration Flags
########################################################################################
USE_WANDB = False  # Set to True to enable Weights & Biases logging
USE_SEED = False  # Set to True to enable reproducibility
RANDOM_SEED = 42  # Seed value for reproducibility
TRAIN_FLAG = True  # Set to True to force retraining the model
USE_TEST = False  # Set to True to visualize test set instead of train set

# Training configuration
EPOCHS = 10  # Number of training epochs
LR = 0.001  # Learning rate
BATCH_SIZE = 20  # Batch size

# File paths
fig_path = "./figures"
model_path = "./models"
run_timestamp = datetime.datetime.now().strftime("%m%d-%H%M")

# WANDB configuration
WANDB_PROJECT = "lab4-cat-classifier"
WANDB_RUN_NAME = f"lab4sol-{run_timestamp}"
data_dir = "/remote_home/Project_Data"
os.makedirs(data_dir, exist_ok=True)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(data_dir, batch_size=BATCH_SIZE):

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
            x_act=int(parts[1])
            y_act=int(parts[3])
            intensity=int(parts[5])
            return [x_act, y_act]

        def __getitem__(self, idx):
            file = self.files[idx]
            path = os.path.join(self.data_dir, file)

            image = np.load(path)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image = image / 255.0

            x, y = self.parse_point_from_filename(file)

            target = torch.tensor([x, y], dtype=torch.float32)

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

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader


class SingleObjectDetector(nn.Module):
    def __init__(self):
        super(SingleObjectDetector, self).__init__()
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Input: RGB image
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 16 -> 8
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64*128*128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output layers
        self.presence = nn.Linear(64, 1)  # Sigmoid later
        self.coords = nn.Linear(64, 2)    # x, y coordinates (normalized)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        presence = torch.sigmoid(self.presence(x))  # 0-1 probability
        coords = self.coords(x)
        return presence, coords


model = SingleObjectDetector()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
trainloader, valloader, testloader = get_dataloaders(data_dir, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, targets in trainloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        presence, coords = model(images)
        
        # MSE loss for coordinates
        loss = criterion(coords, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(trainloader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")
    
    # Optional: evaluation on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in valloader:
            images = images.to(device)
            targets = targets.to(device)
            _, coords = model(images)
            val_loss += criterion(coords, targets).item() * images.size(0)
    val_loss /= len(valloader.dataset)
    print(f"Validation Loss: {val_loss:.6f}")




def detector_loss(pred_presence, pred_coords, true_presence, true_coords):
    # Binary cross entropy for presence
    bce = F.binary_cross_entropy(pred_presence, true_presence)
    
    # Mean squared error for coordinates, only when object is present
    mask = true_presence.bool()  # Only compute coords loss when object exists
    mse = F.mse_loss(pred_coords[mask], true_coords[mask]) if mask.any() else 0.0
    
    return bce + mse

