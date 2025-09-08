import os
from pathlib import Path
from unittest import loader
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import torch.nn as nn
from tqdm import tqdm
import csv
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Config/Constants

DATA_DIR = str(Path(__file__).parent.parent / "data")
IMG_SIZE = 224        
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 50
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
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        
        x = self.features(x)
        x = self.classifier(x)
        
        return x

def loss_and_optimizer(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    return criterion, optimizer

def train_one_epoch(model, loader, criterion, optimizer, device):
    
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc="Training", leave=False, total=len(loader))
    
    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        percent_done = 100.0 * (batch_idx + 1) / len(loader)
        it_s = loop.format_dict['rate'] if 'rate' in loop.format_dict and loop.format_dict['rate'] is not None else 0.0

        loop.set_postfix(
            loss=loss.item(),
            acc=correct / total if total > 0 else 0,
            batch=f"{total}/{len(loader.dataset)}",
            lr=optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else None,
            it_s=f"{it_s:.2f} it/s",
            pct=f"{percent_done:.1f}%",
            refresh=True
        )


    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc="Evaluating", leave=False, total=len(loader))
    
    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        percent_done = 100.0 * (batch_idx + 1) / len(loader)
        it_s = loop.format_dict['rate'] if 'rate' in loop.format_dict and loop.format_dict['rate'] is not None else 0.0

        loop.set_postfix(        
            loss=loss.item(),
            acc=correct / total if total > 0 else 0,
            batch=f"{total}/{len(loader.dataset)}",
            lr=None,
            it_s=f"{it_s:.2f} it/s",
            pct=f"{percent_done:.1f}%",
            refresh=True
        )



    return running_loss / total, correct / total



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

    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}\n")
        import matplotlib.pyplot as plt

        if epoch == 0:
            csv_file = open("training_metrics.csv", mode="w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
            train_losses, train_accs, val_losses, val_accs = [], [], [], []

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        csv_writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
        csv_file.flush()

        if epoch == EPOCHS - 1:

            csv_file.close()

            metrics = {
                "Train Loss": train_losses,
                "Train Acc": train_accs,
                "Val Loss": val_losses,
                "Val Acc": val_accs
            }

            for name, values in metrics.items():

                plt.figure(figsize=(10, 1))
                sns.heatmap([values], annot=True, fmt=".4f", cmap="viridis", cbar=True)
                plt.title(f"{name} Heatmap")
                plt.xlabel("Epoch")
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"{name.replace(' ', '_').lower()}_heatmap.png")
                plt.close()

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            
            torch.save(model.state_dict(), "best_control_model.pth")
            
            print("Best model saved.\n")

        print("Saved new best model")
    

if __name__ == "__main__":

    main()