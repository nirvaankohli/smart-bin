import os
from pathlib import Path
import torch
from torchutils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Config/Constants

DATA_DIR = "/data"       
IMG_SIZE = 224        
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 4     


# Transformations

def transformations():

    train_transformatons = transforms.Compose([

        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Want to augment but since this is the control group, we won't augment
        transforms.ToTensor(),

        transforms.Normalize(

            mean = [0.6731, 0.6398, 0.6048],
            std = [0.1832, 0.1824, 0.1928]

        ),

    ])

    val_transformatons = transforms.Compose([

        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        
        transforms.ToTensor(),

        transforms.Normalize(

            mean = [0.6731, 0.6398, 0.6048],
            std = [0.1832, 0.1824, 0.1928]

        ),

    ])

    return train_transformatons, val_transformatons



def dataset_split_and_loader(transformations_train, transformations_val):

    full_dataset = datasets.ImageFolder(

        root=DATA_DIR,
        transform=transformations_train

    )

    num_classes = len(full_dataset.classes)
    num_samples = len(full_dataset)
    val_size = int(VAL_SPLIT * num_samples)
    train_size = num_samples - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_indeces = val_subset.indices
    train_indeces = train_subset.indices

    train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transformations_train)
    val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transformations_val)

    train_dataset = Subset(train_dataset, train_indeces)
    val_dataset = Subset(val_dataset, val_indeces)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader

class controlCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = torch.nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = torch.nn.Sequential(

            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)

        )

    def forward(self, x):
        
        x = self.features(x)
        x = self.classifier(x)
        
        return x

def loss_and_optimizer(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    return criterion, optimizer

def train_one_epoch(model, train_loader, criterion, optimizer):

    model.train()
    

def main():

    # Get transformations

    print("Preparing data loaders...")

    transformations_train, transformations_val = transformations()

    print("Data loaders ready. \n")

    print("Preparing datasets...")

    # Get data loaders & split dataset

    train_loader, val_loader = dataset_split_and_loader(transformations_train, transformations_val)

    print("Datasets ready. \n")
    print("Preparing model...")

    model = controlCNN(6).to(device)

    print("Model ready. \n")
    print("Preparing Loss + Optimizer...")

    criterion, optimizer = loss_and_optimizer(model)

    print("Loss + Optimizer ready. \n")
    print("Starting training...")
    

if __name__ == "__main__":

    main()