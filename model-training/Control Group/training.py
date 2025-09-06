import os
from pathlib import Path
import torch
from torchutils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

IMG_SIZE = 224

device = "cpu"

train_transformatons = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    
    # Want to augment but since this is the control group, we won't augment

    transforms.ToTensor(),

    transforms.Normalize

])