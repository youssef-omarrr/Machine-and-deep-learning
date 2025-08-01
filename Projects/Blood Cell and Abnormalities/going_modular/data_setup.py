from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path


def create_dataloaders(root:Path,
                        transform: transforms = None,
                        batch_size:int = 32):
    """
    Creates PyTorch DataLoaders for training and testing from an image folder dataset.

    Assumes that `root` contains subdirectories representing class names,
    with images inside each subdirectory.

    The dataset is split into 80% training and 20% testing. If no transform is 
    provided, a default transform is applied which resizes images to 224x224 
    and converts them to tensors.

    Args:
        root (Path): Path to the root directory of the dataset (should follow ImageFolder format).
        transform (torchvision.transforms, optional): A set of transformations to apply to the images.
                                                        If None, a default resize + ToTensor is used.
        batch_size (int): Number of samples per batch to load.

    Returns:
        train_dataloader (DataLoader): DataLoader for the training set.
        test_dataloader (DataLoader): DataLoader for the testing set.
        train_dataset (Dataset): Dataset for the training set.
        test_dataset (Dataset): Dataset for the testing set.
        class_names (list[str]): List of class names inferred from subdirectory names.
    """
    
    # Define a transform that resizes all images to the same shape
    if transform == None:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    # Get dataset using ImageFolder
    dataset = datasets.ImageFolder(
        root= root,
        transform= transform
    )

    # Define train and test lengths (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Set random seed 
    torch.manual_seed(42)
    random.seed(42)

    # Random split dataset into train and test datasets
    train_dataset, test_dataset = random_split(dataset, lengths=[train_size, test_size])

    # Get dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle= False)

    # Get class names
    class_names = dataset.classes
    
    return train_dataloader, test_dataloader, train_dataset, test_dataset, class_names


from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
import torch

def get_x_percent_dataloader(dataset: torch.utils.data.Dataset, 
                            percent: float,
                            seed:int = 0,
                            batch_size:int =32, 
                            shuffle:bool =True) -> DataLoader:
    """
    Creates a DataLoader that contains a specified percentage of samples from each class 
    in the provided dataset.

    This function works with both full datasets (like ImageFolder) and Subsets. It 
    ensures that each class contributes approximately the same percentage of its data.

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.Subset): 
            The dataset from which to sample (e.g., ImageFolder or a Subset).
        percent (float): 
            The percentage of samples to include per class (e.g., 0.1 for 10%).
        batch_size (int, optional): 
            Number of samples per batch in the returned DataLoader. Default is 32.
        shuffle (bool, optional): 
            Whether to shuffle the samples in the DataLoader. Default is True.
        seed (int, optional): 
            Random seed for reproducibility. Default is 42.

    Returns:
        DataLoader: A PyTorch DataLoader containing the selected subset of the dataset.
    """
    if seed != 0:
        random.seed(seed)

    # Get true targets and indices from base dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        base_indices = dataset.indices
        targets = [base_dataset.targets[i] for i in base_indices]
    else:
        base_dataset = dataset
        base_indices = list(range(len(dataset)))
        targets = base_dataset.targets
        

    # Group indices by class label
    class_to_idx = defaultdict(list)
    for base_idx, label in zip(base_indices, targets):
        class_to_idx[label].append(base_idx)

    # Sample x% of each class
    selected_indices = []
    for label, idxs in class_to_idx.items():
        n = max(1, int(percent * len(idxs)))
        selected_indices.extend(random.sample(idxs, n))

    # Create a subset and dataloader
    subset = Subset(base_dataset, selected_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

    return dataloader