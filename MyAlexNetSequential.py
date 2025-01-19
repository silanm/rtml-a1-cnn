# AlexNet Implementation with PyTorch Sequential API with optional LRN (Local Response Normalization)

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy

# Data Preprocessing Pipeline
# - Resize images to 256x256
# - Center crop to 224x224
# - Convert to tensor and normalize with CIFAR-10 mean and std
preprocess_pipeline = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),  # CIFAR-10 mean and std
    ]
)

# Load CIFAR-10 dataset with the preprocessing pipeline
train_dataset_full = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=preprocess_pipeline
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=preprocess_pipeline
)

# Split train dataset into training and validation sets (80:20 split)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [40000, 10000])

# Create data loaders for batching and shuffling
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True, num_workers=2
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=4, shuffle=False, num_workers=2
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=2
)

dataloaders = {"train": train_dataloader, "val": val_dataloader}


# Define the AlexNet architecture using PyTorch Sequential API
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class LocalResponseNorm(nn.Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=2):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        div = x.pow(2).unsqueeze(1)
        div = torch.nn.functional.avg_pool3d(
            div, (self.size, 1, 1), stride=1, padding=(self.size // 2, 0, 0)
        ).squeeze(1)
        div = div.mul(self.alpha).add(self.k).pow(self.beta)
        return x / div


# Initialize the AlexNet model with optional LRN
NUM_CLASSES = 10


# Function to build the model
def build_alexnet_model(with_lrn=False):
    layers = [
        nn.Conv2d(
            3, 96, kernel_size=11, stride=4, padding=2
        ),  # First Conv Layer; 3 input channels, 96 output channels
        nn.ReLU(inplace=True),
    ]
    if with_lrn:
        layers.append(LocalResponseNorm(size=5))  # Add LRN layer after ReLU

    layers.extend(
        [
            nn.MaxPool2d(kernel_size=3, stride=2),  # First Pooling Layer
            nn.Conv2d(
                96, 256, kernel_size=5, padding=2
            ),  # Second Conv Layer; 96 input channels, 256 output channels
            nn.ReLU(inplace=True),
        ]
    )
    if with_lrn:
        layers.append(LocalResponseNorm(size=5))  # Add LRN layer after ReLU

    layers.extend(
        [
            nn.MaxPool2d(kernel_size=3, stride=2),  # Second Pooling Layer
            nn.Conv2d(
                256, 384, kernel_size=3, padding=1
            ),  # Third Conv Layer; 256 input channels, 384 output channels
            nn.ReLU(inplace=True),
            nn.Conv2d(
                384, 384, kernel_size=3, padding=1
            ),  # Fourth Conv Layer; 384 input channels, 384 output channels
            nn.ReLU(inplace=True),
            nn.Conv2d(
                384, 256, kernel_size=3, padding=1
            ),  # Fifth Conv Layer; 384 input channels, 256 output channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Third Pooling Layer
            nn.AdaptiveAvgPool2d((6, 6)),  # Adaptive Pooling for variable input sizes
            Flatten(),  # Flatten layer for fully connected layers
            nn.Dropout(),  # Dropout for regularization; 50% probability
            nn.Linear(
                256 * 6 * 6, 4096
            ),  # First Fully Connected Layer; 256*6*6 input features, 4096 output features
            nn.ReLU(inplace=True),
            nn.Dropout(),  # Dropout; 50% probability
            nn.Linear(
                4096, 4096
            ),  # Second Fully Connected Layer; 4096 input features, 4096 output features
            nn.ReLU(inplace=True),
            nn.Linear(
                4096, NUM_CLASSES
            ),  # Output Layer for classification; 4096 input features, 10 output features
        ]
    )

    return nn.Sequential(*layers)


# MPS/CPU/CUDA (4th GPU)
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda:3" if torch.cuda.is_available() else "cpu"
)


# Training function
def train_model(
    model, dataloaders, criterion, optimizer, num_epochs=25, weights_name="model_weights"
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  # Deep copy of model weights
    best_acc = 0.0  # Initialize best accuracy to 0
    val_acc_history = []
    train_loss_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}\n{'-' * 10}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backpropagation and optimization step only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(
                    preds == labels.data
                )  # Only if the prediction is correct

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset
            )  # Accuracy = Correct Predictions / Total Predictions

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"{weights_name}.pth")

            if phase == "val":
                val_acc_history.append(epoch_acc.item())
            elif phase == "train":
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss_history


# Model training configuration
criterion = nn.CrossEntropyLoss()

# Train two versions of AlexNet (with and without LRN)
alexnet_without_lrn = build_alexnet_model(with_lrn=False).to(device)
alexnet_with_lrn = build_alexnet_model(with_lrn=True).to(device)

optimizer_without_lrn = optim.SGD(alexnet_without_lrn.parameters(), lr=0.001, momentum=0.9)
optimizer_with_lrn = optim.SGD(alexnet_with_lrn.parameters(), lr=0.001, momentum=0.9)

# Train AlexNet without LRN
print("Training AlexNet without LRN...")
best_model_without_lrn, val_acc_history_without_lrn, train_loss_history_without_lrn = train_model(
    alexnet_without_lrn,
    dataloaders,
    criterion,
    optimizer_without_lrn,
    num_epochs=10,
    weights_name="alexnet_without_lrn",
)

# Train AlexNet with LRN
print("Training AlexNet with LRN...")
best_model_with_lrn, val_acc_history_with_lrn, train_loss_history_with_lrn = train_model(
    alexnet_with_lrn,
    dataloaders,
    criterion,
    optimizer_with_lrn,
    num_epochs=10,
    weights_name="alexnet_with_lrn",
)

print("AlexNet training with and without LRN completed.")
