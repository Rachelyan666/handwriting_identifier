import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(dataset="mnist", bs_train=128, bs_test=256, num_workers=2):
    mean, std = (0.1307,), (0.3081,)
    train_tfms = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset.lower() == "mnist":
        train_ds = datasets.MNIST("data", train=True, download=True, transform=train_tfms)
        test_ds  = datasets.MNIST("data", train=False, download=True, transform=test_tfms)
        classes = [str(i) for i in range(10)]
    elif dataset.lower() == "emnist":
        train_ds = datasets.EMNIST("data", split="balanced", train=True, download=True, transform=train_tfms)
        test_ds  = datasets.EMNIST("data", split="balanced", train=False, download=True, transform=test_tfms)
        classes = list(train_ds.classes)
    else:
        raise ValueError("dataset must be 'mnist' or 'emnist'")

    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs_test, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, classes
