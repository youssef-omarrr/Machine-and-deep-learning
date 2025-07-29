import torch
import torchvision
from torch import nn

def create_vit_model(num_classes: int=3,
                    seed: int = 42):
    """Creates a ViT-B/16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of target classes. Defaults to 3.
        seed (int, optional): random seed value for output layer. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViT-B/16 feature extractor model. 
        transforms (torchvision.transforms): ViT-B/16 image transforms.
    """
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights = "DEFAULT")
    
    torch.manual_seed(seed)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    
    return model, transforms
