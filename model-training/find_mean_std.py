from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

DATA_DIR = "data/"

dataset = datasets.ImageFolder(

    root=DATA_DIR,
    transform=transforms.ToTensor()

)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

mean = 0.
std = 0.
nb_samples = 0.

for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("Mean:", mean)
print("Std:", std)