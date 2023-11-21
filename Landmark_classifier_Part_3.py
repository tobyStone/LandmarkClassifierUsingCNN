import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import os

# Define hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
dropout_rate = 0.4
weight_decay = 1e-4
num_classes = 50

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Normalization for validation and test
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load data
data_dir = 'data'  # Replace with your data directory
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
dataset_size = len(dataset)

# Split dataset into train, validation, and test sets
split_indices = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, val_test_indices = next(split_indices.split(range(dataset_size), dataset.targets))

val_test_split = int(0.5 * len(val_test_indices))
val_indices = val_test_indices[:val_test_split]
test_indices = val_test_indices[val_test_split:]

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load a pre-trained model (e.g., ResNet-18) and modify the final layer
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.fc.in_features, num_classes)
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
best_val_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_predictions = []
    val_true_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Save the model if it has the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

# Load the best model and test
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_predictions = []
test_true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_true_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')

