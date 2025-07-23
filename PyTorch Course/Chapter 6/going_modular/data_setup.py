"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os 

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloader( 
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    transform_test: transforms.Compose = None
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.
    transform_test (optional): specific transform for test dataset.
    
    Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
        train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    
    # 1) Use ImageFolder to create dataset(s)
    train_dataset = datasets.ImageFolder(train_dir,
                                        transform= transform,
                                        target_transform= None)

    test_dataset = datasets.ImageFolder(test_dir,
                                        transform= transform_test if transform_test else transform)
    
    # 2) Get class_names
    class_names = train_dataset.classes
    
    # 3) Turn train and test Datasets into DataLoaders
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True)
    
    return train_loader, test_loader, class_names
