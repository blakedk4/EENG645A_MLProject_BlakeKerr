'''1. Architecture idea

Input → Conv layers → Flatten → Fully Connected (FC) layers

Two outputs:

presence: 1 neuron with sigmoid (0–1 probability)

coords: 2 neurons (x, y normalized between 0–1)

We’ll assume images are 64x64 for simplicity.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleObjectDetector(nn.Module):
    def __init__(self):
        super(SingleObjectDetector, self).__init__()
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Input: RGB image
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
            nn.Linear(64*8*8, 128),
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
        coords = torch.sigmoid(self.coords(x))      # 0-1 normalized coordinates
        return presence, coords


def detector_loss(pred_presence, pred_coords, true_presence, true_coords):
    # Binary cross entropy for presence
    bce = F.binary_cross_entropy(pred_presence, true_presence)
    
    # Mean squared error for coordinates, only when object is present
    mask = true_presence.bool()  # Only compute coords loss when object exists
    mse = F.mse_loss(pred_coords[mask], true_coords[mask]) if mask.any() else 0.0
    
    return bce + mse

