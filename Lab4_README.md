[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hflSbVb7)
# Lab 4
We will use the machine learning workflow to solve our problem again. Thus, this lab is broken into the same 7 part workflow to solve our problem
## Problem 
We need to determine if a cat is in a picture. By cat, I mean any kind of cat. We can assume the pictures can be slimmed down to a `32x32` image without any loss of information

In this lab, you will build a Convolutional Neural Network (CNN) to classify images as either containing a cat or not containing a cat. You'll work with the CIFAR-10 dataset and learn to handle class imbalance.

## Learning Objectives
- Build and train a CNN using PyTorch
- Handle imbalanced datasets using class weights
- Evaluate model performance using confusion matrices and classification reports
- Understand the difference between accuracy and recall for imbalanced problems

---

## ACEHUB Environment Setup
- **image**: `git.antcenter.net:4567/nyielding/acehub-pytorch-image:latest`
- **CPU**: 2 cores
- **Memory**: 16 GB
- **GPU**: 1 (recommended for faster training)

---

## Grading Rubric (100 points)

Your submission will be automatically graded based on the following criteria:

| Category | Points | Requirements |
|----------|--------|--------------|
| **Data Pipeline** | 25 | Implement `get_dataloaders()` returning train/val/test DataLoaders, and `modify_targets()` to convert labels to binary (cat=1, not cat=0) |
| **Data Imbalance** | 10 | Verify ~10% of samples are cats after label conversion |
| **Model Architecture** | 15 | Define `ANN` class with Conv2d layers, Linear layers, and correct output shape (2 classes) |
| **Saved Artifacts** | 25 | Save trained model to `models/model.pth` and confusion matrix figures to `figures/` |
| **Model Performance** | 25 | Model must predict both classes (not just "not cat") and achieve ≥30% cat recall |

---

## Step-by-Step Instructions

### Step 1: Get the Data

We will use the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes. The dataset is automatically downloaded using torchvision:

```python
import torchvision
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
```

The 10 classes in CIFAR-10 are:
```python
label_names = {
    0: 'airplane',
    1: 'automobile', 
    2: 'bird',
    3: 'cat',        # This is our target class!
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}
```

**Your task**: Convert this to a binary classification problem (cat vs not cat).

### Step 2: Implement `modify_targets()`

Implement the `modify_targets()` function to convert labels:
- Cat (class 3) → 1
- All other classes → 0

```python
def modify_targets(dataset):
    # dataset.targets is a list of labels
    # Modify it in place to be binary
    pass  # TODO: implement this
```

### Step 3: Prepare the Data

1. **Normalize the images** using `transforms.Normalize()`:
   ```python
   transform = transforms.Compose([
       transforms.ToTensor(),
       # use transforms.Normalize here,
   ])
   ```

2. **Split the training data** into train (80%) and validation (20%) sets:
   ```python
   train_size = int(0.8 * len(trainset))
   val_size = len(trainset) - train_size
   train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
   ```

3. **Create DataLoaders** for train, validation, and test sets.

### Step 4: Build Your CNN Model

Define your model in the `ANN` class. A typical architecture:

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        # Example:
        # some CNN layers
        # Some pooling and activation layers
        # some linear layers
        
    def forward(self, x):
        # Define forward pass
        return x
```

**Requirements**:
- Must have at least one `Conv2d` layer
- Must have at least one `Linear` layer  
- Output must have 2 neurons (not cat, cat)

### Step 5: Handle Class Imbalance

Since only ~10% of images are cats, the model may learn to always predict "not cat" and achieve 90% accuracy! To fix this, use **class weights**:

```python
# Calculate weights inversely proportional to class frequency
n_cats = 5000  # ~10% of 50,000 training images
n_not_cats = 45000
class_weights = None  # calculate this into a torch.tensor

# Don't forget your loss function
criterion = None
```

### Step 6: Train Your Model

Implement the training loop in `train_model()`:
1. Forward pass
2. Compute loss
3. Backward pass
4. Update weights
5. Validate using `evaluate_model()`

### Step 7: Evaluate and Save

1. **Use `evaluate_model()`** to get predictions on test data
2. **Print classification report**:
   ```python
   print(sklearn.metrics.classification_report(y_true, y_pred, target_names=["not cat", "cat"]))
   ```
3. **Generate confusion matrices** using `plot_cm()` - save to `figures/`
4. **Save your model**:
   ```python
   torch.save(model.state_dict(), "models/model.pth")
   ```

---

## Key Functions to Implement

| Function | Description |
|----------|-------------|
| `modify_targets(dataset)` | Convert labels to binary (cat=1, others=0) |
| `get_dataloaders()` | Return train, val, test DataLoaders |
| `ANN` class | Define CNN architecture |
| `evaluate_model()` | Evaluate model on a dataloader (partially implemented) |
| `train_model()` | Training loop with validation |

---

## Expected Output Files

Your submission must include:

```
models/
  └── model.pth          # Trained model weights
figures/
  ├── confusion_matrix.png
  └── confusion_matrix_normalized.png
```

---

## Tips for Success

1. **Start simple**: Get a basic CNN working before adding complexity
2. **Check your labels**: Print some labels after `modify_targets()` to verify they're binary
3. **Use class weights**: Without them, your model will likely have 0% cat recall
4. **Monitor validation loss**: Stop training if validation loss increases (overfitting)
5. **Target cat recall ≥ 30%**: The autograder checks that your model can actually detect cats

---

## Analysis Questions

Answer these in comments in your code:

1. **What would be the accuracy if we always predicted "not cat"?**
   - (Hint: What percentage of the dataset is not cats?)

2. **What was your cat recall before using class weights?**

3. **What was your cat recall after using class weights?**

---

## Resources

- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Handling Imbalanced Data](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
   
