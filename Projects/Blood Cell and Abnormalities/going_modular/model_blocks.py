# -------------------------------
# Imports and Hyperparameters
# -------------------------------
import torch
import math

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

# -------------------------------
# Patch Embedding
# -------------------------------
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    def __init__(self, in_channels: int = HYPER_PARAMS['color_channels'],
                        patch_size: int = HYPER_PARAMS['patch_size'],
                        embedding_dim: int = HYPER_PARAMS['embedding_dim']):
        super().__init__()

        self.patch_size = patch_size  # Store patch size for validation during forward pass

        # Step 1: Create a convolutional layer to divide the image into non-overlapping patches
        # Each patch is projected to a vector of size 'embedding_dim'
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                out_channels=embedding_dim,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0)

        # Step 2: Flatten the spatial dimensions (height and width) of the output from the patcher
        # This turns the image into a sequence of patch embeddings
        self.flatten = nn.Flatten(start_dim=2, end_dim=3) # [batch_size, embedding_dim, num_patches_h, num_patches_w]

    def forward(self, x):
        # Step 3: Get the image resolution (assuming square images)
        img_resolution = x.shape[-1]

        # Step 4: Ensure the image size is divisible by the patch size (no partial patches)
        assert img_resolution % self.patch_size == 0, \
            f"Input image size must be divisible by patch size, image shape: {img_resolution}, patch size: {self.patch_size}"

        # Step 5: Apply the patch embedding conv layer and flatten the result
        x_output = self.flatten(self.patcher(x))

        # print(x_output.shape)  # Debugging: print shape after patching and flattening

        # Step 6: Rearrange dimensions from [batch, embedding_dim, num_patches]
        # to [batch, num_patches, embedding_dim] to match expected input format
        return x_output.permute(0, 2, 1)
    
    
# -------------------------------
# Class Token & Positional Embedding
# -------------------------------

# Creating a single trainable parameter to act as the class token
class_token = nn.Parameter(torch.randn(1, 1, HYPER_PARAMS['embedding_dim']), 
                            requires_grad= True)

# Create position embedding
position_embedding = nn.Parameter( # Same shape as the output of concating the cass token and the patch embedding 
                                torch.randn(1, HYPER_PARAMS['number_of_patches']+1, HYPER_PARAMS['embedding_dim']),
                                requires_grad= True)

# -------------------------------
# Multi-Head Self Attention (MSA)
# -------------------------------
class MSA(nn.Module):
    """
    Multi-Head Self-Attention (MSA) module.
    
    Args:
        hidden_dim (int): Input and output dimension (usually equal to embedding dim).
        num_heads (int): Number of attention heads to split input into.
    """
    def __init__(self, 
                hidden_dim: int = HYPER_PARAMS['embedding_dim'],
                num_heads: int = HYPER_PARAMS['num_heads']):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.head_dim = hidden_dim // num_heads  # Dimension per head

        # Linear layers to project input into queries, keys, and values
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        # Final linear layer to recombine the heads
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.softmax = nn.Softmax(dim=-1)  # Softmax over attention scores

    def forward(self, X):
        """
        Args:
            X (Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
        Returns:
            Tensor: Output tensor after attention, same shape as input
        """
        B, N, D = X.shape  # Batch, Sequence Length (number of patches + 1), Embedding Dim
        
        # Step 1: Linear projections -> [B, N, D]
        Q = self.q_linear(X)
        K = self.k_linear(X)
        V = self.v_linear(X)

        # Step 2: Split heads -> [B, num_heads, N, head_dim] (for parallel attention)
        Q = Q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Step 3: Attention scores -> [B, num_heads, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 4: Apply softmax to get attention weights
        attn_weights = self.softmax(attn_scores)

        # Step 5: Weighted sum of values -> [B, num_heads, N, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Step 6: Concatenate heads -> [B, N, D]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, D)

        # Step 7: Final linear projection
        output = self.out_linear(attn_output)

        return output
    
# -------------------------------
# MLP Block
# -------------------------------
    
class MLP(nn.Module):
    """
    Feedforward network used in Transformer encoder blocks.

    Args:
        embedding_dim (int): Input and output feature size.
        mlp_size (int): Hidden layer size (expansion size).
        dropout (float): Dropout probability.
    """
    def __init__(self,
                embedding_dim: int = HYPER_PARAMS['embedding_dim'],
                mlp_size: int = HYPER_PARAMS['MLP_size'],
                dropout: float = 0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),  # expand
            nn.GELU(),                                                     # non-linearity
            nn.Dropout(p=dropout),                                         # regularization
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),  # project back
            nn.Dropout(p=dropout)                                          # regularization
        )

    def forward(self, x):
        return self.mlp(x)

# -------------------------------
# Encoder Block
# -------------------------------
    
class Encode(nn.Module):
    """
    Transformer Encoder Block used in Vision Transformer (ViT).

    Args:
        MSA (nn.Module): Multi-head self-attention module constructor.
        MLP (nn.Module): Feedforward MLP block constructor.
        hidden_dim (int): Embedding dimension of the input.
        num_heads (int): Number of attention heads.
        mlp_size (int): Hidden layer size of the MLP.
        mlp_dropout (float): Dropout rate for the MLP.
    """
    def __init__(self,
                MSA: nn.Module,
                MLP: nn.Module,
                
                hidden_dim: int = HYPER_PARAMS['embedding_dim'],
                num_heads: int = HYPER_PARAMS['num_heads'],
                
                mlp_size: int = HYPER_PARAMS['MLP_size'],
                mlp_dropout: float = 0.1):
        super().__init__()

        self.msa = MSA(hidden_dim, num_heads)
        self.mlp = MLP(hidden_dim, mlp_size, dropout=mlp_dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def forward(self, x):
        x = self.msa(self.layer_norm(x)) + x  # Attention + Residual
        x = self.mlp(self.layer_norm(x)) + x  # MLP + Residual
        return x
    
    
# -------------------------------
# Classification Head
# -------------------------------

# Classification head for ViT
MLP_Head = nn.Sequential(
    # Normalize the embedding output from the encoder
    nn.LayerNorm(normalized_shape=HYPER_PARAMS['embedding_dim']),
    
    # Project to number of classes for classification
    nn.Linear(
        in_features=HYPER_PARAMS['embedding_dim'],
        out_features=HYPER_PARAMS['num_classes']
    )
)
