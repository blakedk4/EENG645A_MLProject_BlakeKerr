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
import matplotlib.pyplot as plt

from utils.utils import (
    init_wandb,
    log_to_wandb,
    log_image_to_wandb,
    finish_wandb,
    plot_cm,
    set_seed,
)
from tqdm import tqdm
# Hyperparameter optimization
import optuna
from optuna_integration.wandb import WeightsAndBiasesCallback

USE_WANDB = False  # Set to True to enable Weights & Biases logging
USE_SEED = False  # Set to True to enable reproducibility
RANDOM_SEED = 42  # Seed value for reproducibility
TRAIN_FLAG = False  # Set to True to force retraining the model
USE_TEST = False  # Set to True to visualize test set instead of train set
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configuration
EPOCHS = 20  # Number of training epochs
LR = 0.001  # Learning rate
BATCH_SIZE = 2  # Batch size
GRID_SIZE = 16  # number of subdivisions per side (N x N grid)
TARGET_SIZE = 512  # Keep original size
COORD_WEIGHT = 100

# File paths
fig_path = "./figures"
model_path = "./models"
run_timestamp = datetime.datetime.now().strftime("%m%d-%H%M")

# WANDB configuration
WANDB_PROJECT = "lab4-cat-classifier"
WANDB_RUN_NAME = f"lab4sol-{run_timestamp}"
data_dir = "/remote_home/Project_Data"
os.makedirs(data_dir, exist_ok=True)
NUM_SAMPLES = 20
USE_OPTUNA = False


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
    #print(f"Total images: {total}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
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

def evaluate_model(model, dataloader, threshold_pixels=10, original_size=(1024, 1024)):
    print("Eval Loop")

    model.eval()

    all_probs = []
    all_labels = []

    all_pred_pixels = []
    all_true_pixels = []

    orig_h, orig_w = original_size

    with torch.no_grad():
        for images, coords_target, presence_target in dataloader:

            images = images.to(DEVICE)
            coords_target = coords_target.to(DEVICE)
            presence_target = presence_target.to(DEVICE)

            coords_pred, presence_logits = model(images)

            probs = torch.sigmoid(presence_logits)

            # --- ROC COLLECTION ---
            # Max presence probability per image
            #image_probs = probs.max(dim=1).values

            # Does this image contain any object?
            #image_labels = presence_target.max(dim=1).values

            # --- ROC COLLECTION ---
            all_probs.extend(probs.detach().cpu().numpy().flatten())
            all_labels.extend(presence_target.detach().cpu().numpy().flatten())

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

def train_model(model, trainloader, valloader, criterion, optimizer, epochs, use_wandb=False):
    model.to(DEVICE)
    history = {"train_loss": [], "val_loss": []}
    coord_loss_fn = nn.SmoothL1Loss(reduction="none")
    presence_loss_fn = nn.BCEWithLogitsLoss()
    #for epoch in range(epochs):
    model.train()
    for epoch in range(0,epochs):
        running_loss = 0.0
        for images, coords_target, presence_target in trainloader:
            images = images.to(DEVICE)
            coords_target = coords_target.to(DEVICE)
            presence_target = presence_target.to(DEVICE)
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

            loss = loss_presence+COORD_WEIGHT*coord_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        history["train_loss"].append(train_loss)
        val_results = evaluate_model(model, valloader)
        val_accuracy = val_results["accuracy"]
        #if use_wandb:
            #log_to_wandb({"train_loss": train_loss, "val_loss": val_loss}, step=epoch, use_wandb=use_wandb)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    print("Finished Training")
    return model, history


# ============================================================================
# STRATEGY 1: OPTUNA STANDALONE SEARCH
# ============================================================================
def optuna_objective(trial):

    # Hyperparameters to search
    lr=LR
    #lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    coord_weight = trial.suggest_categorical("coord_weight", [10,25,50,100,200,400])
    grid_size = trial.suggest_categorical("grid_size", [8, 16, 32])
    target_size = trial.suggest_categorical("target_size",[256, 384, 512])
    # Load data
    trainloader, valloader, _ = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        target_size=(TARGET_SIZE, TARGET_SIZE)
    )

    model = SingleObjectDetector().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    coord_loss_fn = nn.SmoothL1Loss(reduction="none")
    presence_loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        for images, coords_target, presence_target in trainloader:

            images = images.to(DEVICE)
            coords_target = coords_target.to(DEVICE)
            presence_target = presence_target.to(DEVICE)

            optimizer.zero_grad()

            coords_pred, presence_logits = model(images)

            loss_presence = presence_loss_fn(
                presence_logits,
                presence_target
            )

            coord_loss_raw = coord_loss_fn(coords_pred, coords_target)
            mask = presence_target.unsqueeze(-1)

            coord_loss = (coord_loss_raw * mask).sum() / mask.sum().clamp(min=1)

            loss = loss_presence + coord_weight * coord_loss

            loss.backward()
            optimizer.step()

    # Evaluate
    results = evaluate_model(model, valloader)

    return results["accuracy"]

def run_optuna_search(n_trials=NUM_SAMPLES):

    print("\n" + "=" * 60)
    print("OPTUNA SEARCH")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=3
        )
    )

    study.optimize(optuna_objective, n_trials=n_trials)

    print("\nBest accuracy:", study.best_trial.value)
    print("Best hyperparameters:")

    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")

    return study

def main():
    if USE_SEED:
        set_seed(RANDOM_SEED)
    print("DEVICE: ", DEVICE)
    if "cuda" in str(DEVICE):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING Using CPU (this will be slow!)")

    fig_path = "./figures"
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    trainloader, valloader, testloader = get_dataloaders(data_dir, batch_size=BATCH_SIZE, target_size=(TARGET_SIZE, TARGET_SIZE))

    config = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "seed": RANDOM_SEED if USE_SEED else None}
    init_wandb(project_name=WANDB_PROJECT, run_name=WANDB_RUN_NAME, config=config, use_wandb=USE_WANDB)

    model_save_path = f"{model_path}/single_object_detector.pth"
    model = SingleObjectDetector().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )
    print("Begin Training")
    if USE_OPTUNA:

        study = run_optuna_search(NUM_SAMPLES)

        print("\nBest trial:")
        print(f"Accuracy: {study.best_trial.value:.2f}")
        print("Params:", study.best_trial.params)

    else:
        if not os.path.exists(model_save_path) or TRAIN_FLAG:
            model, history = train_model(model, trainloader, valloader, criterion, optimizer, epochs=EPOCHS, use_wandb=USE_WANDB)
            torch.save(model.state_dict(), model_save_path)
        else:
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
        
        loader = testloader if USE_TEST else trainloader
        results = evaluate_model(model, loader)

        y_pred_pixels = np.round(results["pred_pixels"])
        y_true_pixels = results["true_pixels"]
        accuracy = results["accuracy"]

        fpr_cnn, tpr_cnn, _ = roc_curve(results["labels"], results["probs"])
        roc_auc = auc(fpr_cnn, tpr_cnn)

        print("Pixel Accuracy:", results["accuracy"])
        #print("ROC AUC:", roc_auc)
        # Print a few predictions vs true values
        print("\nSample predictions vs true coordinates:")
        for i in range(min(5, len(y_pred_pixels))):
            print(f"Predicted: {y_pred_pixels[i]}, True: {y_true_pixels[i]}")

        errors = np.linalg.norm(y_pred_pixels - y_true_pixels, axis=1)
        print(f"\nMean pixel error: {np.mean(errors):.2f}, Max error: {np.max(errors):.2f}, Accuracy: {results['accuracy']:.4f}")

        finish_wandb(use_wandb=USE_WANDB)
        '''
        plt.figure() 
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})') 
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        plt.title('ROC Curve for Presence Detection') 
        plt.legend(loc="lower right") 
        
        
        roc_save_path = os.path.join(fig_path, "roc_curve.png")
        plt.savefig(roc_save_path,dpi=300,bbox_inches="tight")
        print("Roc Curve Saved")
        plt.close()
        plt.figure(figsize=(7,6))'''
        labels_thresh = np.load("./figures/labels.npy")
        scores_thresh = np.load("./figures/scores.npy")
        #scores_norm = scores_thresh / 6101.0
        #scores_norm = np.clip(scores_norm, 0, 1)  # ensure within [0,1]
        accuracy_thresh = np.load("./figures/accuracy.npy")
        print("Pixel Accuracy:", results["accuracy"])
        print("Threshold Accuracy:", accuracy_thresh*100)
        fpr_thresh, tpr_thresh, _ = roc_curve(labels_thresh, scores_thresh)
        roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
        print("ROC AUC CNN:", roc_auc)
        print("ROC  THRESH:", roc_auc_thresh)
        plt.plot(fpr_cnn, tpr_cnn, color='green', lw=2, label=f'CNN Detector (AUC={roc_auc:.2f})')
        plt.plot(fpr_thresh, tpr_thresh, color='orange', lw=2, label=f'Threshold Detector (AUC={roc_auc_thresh:.2f})')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Comparison: CNN vs Threshold Detector')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

        fig_path = "./figures"
        roc_save_path = os.path.join(fig_path, "roc_curve_Both.png")
        plt.savefig(roc_save_path,dpi=300,bbox_inches="tight")
        print("Roc Curve Saved")
        plt.close()


if __name__ == "__main__":
    main()