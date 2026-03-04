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
from sklearn.metrics import roc_curve, auc


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
LR = 0.0005  # Learning rate
BATCH_SIZE = 8  # Batch size
GRID_SIZE = 16  # number of subdivisions per side (N x N grid)
TARGET_SIZE = 512  # Keep original size

# File paths
fig_path = "./figures"
model_path = "./models"
run_timestamp = datetime.datetime.now().strftime("%m%d-%H%M")

# WANDB configuration
WANDB_PROJECT = "lab4-cat-classifier"
WANDB_RUN_NAME = f"lab4sol-{run_timestamp}"
data_dir = "/remote_home/Project_Data"
os.makedirs(data_dir, exist_ok=True)


def get_dataloaders(data_dir, batch_size=BATCH_SIZE, target_size=(TARGET_SIZE,TARGET_SIZE)):
    class NPYKeypointDataset(Dataset):
        def __init__(self, data_dir, target_size=(TARGET_SIZE,TARGET_SIZE)):
            self.data_dir = data_dir
            self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
            self.target_size = target_size  # store target_size

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
            image = np.load(path).astype(np.float32)

            # Resize only if target_size is provided
            if self.target_size is not None:
                image = cv2.resize(image, self.target_size)

            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

            x, y = self.parse_point_from_filename(file)

            # Normalize to 0-1 global coords
            x_norm = x / 1024
            y_norm = y / 1024

            # --- NEW: Determine which grid cell ---
            col = int(x_norm * GRID_SIZE)
            row = int(y_norm * GRID_SIZE)
            cell_idx = row * GRID_SIZE + col

            # Local coordinates within the cell
            x_local = x_norm * GRID_SIZE - col
            y_local = y_norm * GRID_SIZE - row

            # Create targets
            coords_target = torch.zeros((GRID_SIZE * GRID_SIZE, 2))
            presence_target = torch.zeros(GRID_SIZE * GRID_SIZE)

            coords_target[cell_idx] = torch.tensor([x_local, y_local])
            presence_target[cell_idx] = 1.0

            return image, coords_target, presence_target

    full_dataset = NPYKeypointDataset(data_dir,target_size=(TARGET_SIZE,TARGET_SIZE))

    # 70 / 15 / 15 split
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    print(f"Total images: {total}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
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
        # Output layer (coords only — presence removed)
        self.grid_pool = nn.AdaptiveAvgPool2d((GRID_SIZE, GRID_SIZE))
        self.head = nn.Linear(128, 3)  # per cell output stays same

    '''def forward(self, x):
        x = self.conv_layers(x)
        # Flatten full spatial map
        x = torch.flatten(x, 1)  # shape: [batch, 131072]
        x = self.fc(x)
        coords = self.coords(x)
        presence_logit = self.presence(x)

        return coords, presence_logit'''
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.grid_pool(x)  # [B, 128, GRID_SIZE, GRID_SIZE]
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(B, GRID_SIZE*GRID_SIZE, 128)
        out = self.head(x)
        presence_logits = out[:, :, 0]
        coords = torch.sigmoid(out[:, :, 1:])
        return coords, presence_logits

def evaluate_model(model, dataloader, device, threshold_pixels=10, original_size=(1024, 1024)):
    print("Eval Loop")

    model.eval()

    all_probs = []
    all_labels = []

    all_pred_pixels = []
    all_true_pixels = []

    orig_h, orig_w = original_size

    with torch.no_grad():
        for images, coords_target, presence_target in dataloader:

            images = images.to(device)
            coords_target = coords_target.to(device)
            presence_target = presence_target.to(device)

            coords_pred, presence_logits = model(images)

            probs = torch.sigmoid(presence_logits)

            # --- ROC COLLECTION ---
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(presence_target.cpu().numpy().flatten())

            # --- NEW: Convert grid predictions back to global coords ---
            B = images.size(0)

            for b in range(B):

                # True quadrant
                true_quad = torch.argmax(presence_target[b]).item()

                # Predicted quadrant (highest presence probability)
                pred_quad = torch.argmax(probs[b]).item()

                # Get local coords
                pred_local = coords_pred[b, pred_quad]
                true_local = coords_target[b, true_quad]

                # Convert quadrant index to row/col
                pred_row = pred_quad // GRID_SIZE
                pred_col = pred_quad % GRID_SIZE
                true_row = true_quad // GRID_SIZE
                true_col = true_quad % GRID_SIZE

                # Convert back to global normalized coords
                pred_x_global = (pred_local[0] / GRID_SIZE + 1.0 / GRID_SIZE * pred_col)
                pred_y_global = (pred_local[1] / GRID_SIZE + 1.0 / GRID_SIZE * pred_row)
                true_x_global = (true_local[0] / GRID_SIZE + 1.0 / GRID_SIZE * true_col)
                true_y_global = (true_local[1] / GRID_SIZE + 1.0 / GRID_SIZE * true_row)

                # Convert to pixel space
                pred_x_pixel = pred_x_global.item() * orig_w
                pred_y_pixel = pred_y_global.item() * orig_h

                true_x_pixel = true_x_global.item() * orig_w
                true_y_pixel = true_y_global.item() * orig_h

                all_pred_pixels.append([pred_x_pixel, pred_y_pixel])
                all_true_pixels.append([true_x_pixel, true_y_pixel])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    all_pred_pixels = np.array(all_pred_pixels)
    all_true_pixels = np.array(all_true_pixels)

    # --- ORIGINAL ACCURACY METRIC (Pixel Distance) ---
    distances = np.linalg.norm(all_pred_pixels - all_true_pixels, axis=1)
    correct = np.sum(distances <= threshold_pixels)
    accuracy = correct / len(distances) * 100

    return {
        "probs": all_probs,
        "labels": all_labels,
        "pred_pixels": all_pred_pixels,
        "true_pixels": all_true_pixels,
        "accuracy": accuracy
    }

def train_model(model, trainloader, valloader, criterion, optimizer, device, epochs, use_wandb=False):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    coord_loss_fn = nn.SmoothL1Loss(reduction="none")
    presence_loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, coords_target, presence_target in trainloader:
            images = images.to(device)
            coords_target = coords_target.to(device)
            presence_target = presence_target.to(device)
            optimizer.zero_grad()
            coords_pred,presence_logits = model(images)
            
            # --- NEW: Presence loss ---
            loss_presence = presence_loss_fn(
                presence_logits, presence_target
            )

            # --- NEW: Masked coordinate loss ---
            coord_loss_raw = coord_loss_fn(coords_pred, coords_target)
            mask = presence_target.unsqueeze(-1)
            coord_loss = (coord_loss_raw * mask).sum() / mask.sum().clamp(min=1)

            loss = loss_presence+100*coord_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        history["train_loss"].append(train_loss)
        val_results = evaluate_model(model, valloader, device)
        val_accuracy = val_results["accuracy"]
        #if use_wandb:
            #log_to_wandb({"train_loss": train_loss, "val_loss": val_loss}, step=epoch, use_wandb=use_wandb)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Accuracy: {val_accuracy:.4f}")

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

    trainloader, valloader, testloader = get_dataloaders(data_dir, batch_size=BATCH_SIZE, target_size=(TARGET_SIZE, TARGET_SIZE))

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
    print("Begin Training")
    if not os.path.exists(model_save_path) or TRAIN_FLAG:
        model, history = train_model(model, trainloader, valloader, criterion, optimizer, device, epochs=EPOCHS, use_wandb=USE_WANDB)
        torch.save(model.state_dict(), model_save_path)
    else:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

    loader = testloader if USE_TEST else trainloader
    results = evaluate_model(model, loader, device)

    y_pred_pixels = np.round(results["pred_pixels"])
    y_true_pixels = results["true_pixels"]
    accuracy = results["accuracy"]

    fpr, tpr, _ = roc_curve(results["labels"], results["probs"])
    roc_auc = auc(fpr, tpr)

    print("Pixel Accuracy:", results["accuracy"])
    print("ROC AUC:", roc_auc)
    # Print a few predictions vs true values
    print("\nSample predictions vs true coordinates:")
    for i in range(min(5, len(y_pred_pixels))):
        print(f"Predicted: {y_pred_pixels[i]}, True: {y_true_pixels[i]}")

    errors = np.linalg.norm(y_pred_pixels - y_true_pixels, axis=1)
    print(f"\nMean pixel error: {np.mean(errors):.2f}, Max error: {np.max(errors):.2f}, Accuracy: {results['accuracy']:.4f}")

    finish_wandb(use_wandb=USE_WANDB)


if __name__ == "__main__":
    main()