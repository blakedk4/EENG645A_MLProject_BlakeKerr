"""Lab 4 TEMPLATE"""

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
BATCH_SIZE = 64  # Batch size

# File paths
fig_path = "./figures"
model_path = "./models"
run_timestamp = datetime.datetime.now().strftime("%m%d-%H%M")

# WANDB configuration
WANDB_PROJECT = "lab4-cat-classifier"
WANDB_RUN_NAME = f"lab4sol-{run_timestamp}"


def get_dataloaders(cat_indices, batch_size=64):
    label_names = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }


    # TO.DO: Define transforms to normalize data
    # Use the transforms.Normalize here, not just ToTensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ]
    )

    # Download and load training data
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Download and load test data
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # TO.DO: modify labels to be binary (cat vs not cat)
    def modify_targets(dataset):
        """
        Student must write this function to make labels binary.
        I.e.: cat class (3) -> 1, all other classes -> 0
        Hint: dataset.targets is the list of labels

        Args:
            dataset: torchvision dataset object
        Returns:
            None (modifies dataset in place)
        """
        dataset.targets=[1 if y==3 else 0 for y in dataset.targets]
        pass  # TO.DO: write this function

    # TO.DO: Call the function to modify labels
    modify_targets(trainset)
    modify_targets(testset)

    # TO.DO: Create DataLoaders
    # The assignment asks for train/val/test split.
    # Split trainset into train and validation if needed.
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    split=int(0.8*len(trainset))

    trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False)

    return trainloader, valloader, testloader


# TO.DO: Define your model structure here
class ANN(nn.Module):
    """Don't change the class name from ANN."""

    def __init__(self):
        super(ANN, self).__init__()
        # Define layers
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.fc_layers=nn.Sequential(
            nn.Linear(1152,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self, x):
        # Define forward pass
        x=self.conv_layers(x)
        x=torch.flatten(x,1)
        x=self.fc_layers(x)
        return x


def evaluate_model(model, dataloader, device, criterion):
    """
    Evaluate a model on a given dataloader.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on (cpu/cuda)
        criterion: Loss function (optional, if provided will compute loss)

    Returns:
        dict with keys:
            - 'predictions': numpy array of predicted labels
            - 'true_labels': numpy array of true labels
            - 'accuracy': float, accuracy score
            - 'loss': float, average loss (only if criterion provided)
    """
    model.eval()
    y_pred = []
    y_true = []
    total_loss = 0.0
    correct = 0
    total = 0

    # TO.DO: write the rest of this function
    device="cuda"
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            predicted = None
            images=images.to(device)
            labels=labels.to(device)
            #labels=(labels==3).long()
            outputs=model(images)
            print(outputs.shape,labels.shape,labels.dtype)
            loss=criterion(outputs,labels)
            total_loss+=loss.item()

            _, predicted = torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    results = {
        'predictions': np.array(y_pred),
        'true_labels': np.array(y_true),
        'accuracy': correct / total,
        'loss': total_loss / len(dataloader)
    }

    return results


def train_model(
    model,
    trainloader,
    valloader,
    epochs=10,
    lr=0.001,
    class_weights=None,
    device="cuda",
    use_wandb=False,
):
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # TO.DO: Define loss function, consider class_weights if provided
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # TO.DO: Define optimizer

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # TO.DO: Iterate over trainloader
        # for i, data in enumerate(trainloader, 0):
        #     Don't forget to move data to device
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate metrics
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        # TO.DO: Calculate epoch metrics
        train_loss = running_loss/total
        train_acc = correct/total

        # TO.DO: Validation using evaluate_model
        # Example:
        # val_results = evaluate_model(model, valloader, device, criterion)
        # val_loss = val_results['loss']
        # val_acc = val_results['accuracy']
        val_results = evaluate_model(model, valloader, device, criterion)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Log to wandb
        if use_wandb:
            log_to_wandb(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
                step=epoch,
                use_wandb=use_wandb,
            )

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f},"
            f" Val Acc: {100 * val_acc:.2f}%"
        )

    print("Finished Training")
    return model, history


def main():
    # Set random seed for reproducibility
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

    # Create directories if they don't exist
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    cat_indices = [3]

    trainloader, valloader, testloader = get_dataloaders(cat_indices, BATCH_SIZE)

    # TO.DO: Calculate class weights for imbalance
    class_weights = None
    #class_weights = torch.tensor(class_weights)  # e.g., torch.tensor([weight_for_not_cat, weight_for_cat])

    # Initialize wandb if enabled
    config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": RANDOM_SEED if USE_SEED else None,
    }
    init_wandb(
        project_name=WANDB_PROJECT,
        run_name=WANDB_RUN_NAME,
        config=config,
        use_wandb=USE_WANDB,
    )

    model_save_path = f"{model_path}/model.pth"

    if not os.path.exists(model_save_path) or TRAIN_FLAG:
        # Initialize model
        model = ANN().to(device)
        num_epochs=10
        # TO.DO: Train your model.
        # Only save your final model after training/tuning.
        if class_weights is not None:
            class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # TO.DO: Define loss function, consider class_weights if provided
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in trainloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        # Save your model
        torch.save(model.state_dict(), model_save_path)
    else:
        # load your model
        model = ANN().to(device)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

    # get visualization data
    if USE_TEST:
        loader = testloader
    else:
        loader = trainloader

    # TO.DO: Generate predictions using evaluate_model
    # Use the evaluate_model function defined above
    results = evaluate_model(model,testloader,device,criterion)
    y_pred=results['predictions']
    y_true=results['true_labels']
    class_names = ["not cat", "cat"]

    # make the stats and confusion matrix
    print(
        sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)
    )

    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix using utility function
    plot_cm(
        confusion_matrix,
        classes=class_names,
        title="Confusion matrix, without normalization",
        save_path=f"{fig_path}/confusion_matrix.png",
    )

    # Log to wandb
    if USE_WANDB:
        log_image_to_wandb(
            f"{fig_path}/confusion_matrix.png",
            caption="Confusion Matrix",
            use_wandb=USE_WANDB,
        )

    # Plot normalized confusion matrix using utility function
    plot_cm(
        confusion_matrix,
        classes=class_names,
        normalize=True,
        title="Normalized confusion matrix",
        save_path=f"{fig_path}/confusion_matrix_normalized.png",
    )

    # Log to wandb
    if USE_WANDB:
        log_image_to_wandb(
            f"{fig_path}/confusion_matrix_normalized.png",
            caption="Normalized Confusion Matrix",
            use_wandb=USE_WANDB,
        )

    # Finish wandb run
    finish_wandb(use_wandb=USE_WANDB)


if __name__ == "__main__":
    main()
