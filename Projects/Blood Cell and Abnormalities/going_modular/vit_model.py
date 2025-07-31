import torchvision
from torchvision.models import vit_b_16
from torch import nn  

# First let's init all hyper parameters
HYPER_PARAMS = {"batch_size": 32, 
                "height": 224, # H
                "width" : 224, # W
                "color_channels": 3, # C
                
                "patch_size": 16, # P
                "number_of_patches": 224*224 // 16**2, # N = H*W/P^2
                "embedding_dim": 768, # D = N*(P^2 *C)
                
                "MLP_size": 3072,
                "num_heads": 12,
                
                "num_classes": 14
}

# Define a function to create a Vision Transformer (ViT) model for image classification
def create_ViT(embbeding_dim:int = HYPER_PARAMS['embedding_dim'],
                num_classes: int = HYPER_PARAMS['num_classes']):
    """
    Creates a Vision Transformer (ViT-B/16) model with a custom classification head 
    and returns it along with the appropriate image transformations.

    The model uses pretrained weights from torchvision and freezes all layers 
    except for the classification head, which is replaced to match the specified 
    number of output classes.

    Args:
        embbeding_dim (int): Size of patch embeddings 
                            Defaults to HYPER_PARAMS['embedding_dim'].
        num_classes (int): The number of output classes for the classification task. 
                        Defaults to HYPER_PARAMS['num_classes'].

    Returns:
        vit (torchvision.models.VisionTransformer): The modified Vision Transformer model.
        transforms (Callable): The image preprocessing transformations required by the pretrained model.
    """
    
    # Load the default pretrained weights for ViT-B/16
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    
    # Get the corresponding image transforms (normalization, resizing, etc.)
    transforms = weights.transforms()
    
    # Load the ViT-B/16 model with pretrained weights
    vit = vit_b_16(weights="DEFAULT")
    
    # Freeze all parameters so that only the classification head is trainable
    for param in vit.parameters():
        param.requires_grad = False
    
    # Replace the original classification head with a new one for our number of classes
    vit.heads = nn.Sequential(
        nn.Linear(
            in_features=embbeding_dim,  
            out_features=num_classes                  
        )
    )
    
    # Return the modified model and the required image transforms
    return vit, transforms