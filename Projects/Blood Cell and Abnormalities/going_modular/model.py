import torch
from torch import nn
from .model_blocks import *

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT-B/16) compatible with torchvision's vit_b_16 pretrained weights.
    
    Args:
        image_size: Input image height/width (assumes square images).
        patch_size: Size of each image patch (e.g. 16).
        in_channels: Number of image channels (e.g. 3 for RGB).
        num_classes: Output classes for classification.
        embedding_dim: Size of patch embeddings (e.g. 768).
        depth: Number of transformer encoder blocks (12 for ViT-B/16).
        num_heads: Number of attention heads (12 for ViT-B/16).
        mlp_size: Hidden size of MLP inside encoder blocks (e.g. 3072).
        dropout: Dropout rate.
    """

    def __init__(self,
                image_size: int = HYPER_PARAMS['height'],
                patch_size: int = HYPER_PARAMS['patch_size'],
                in_channels: int = HYPER_PARAMS['color_channels'],
                num_classes: int = HYPER_PARAMS['num_classes'],
                embedding_dim: int = HYPER_PARAMS['embedding_dim'],
                depth: int = 12,
                num_heads: int = HYPER_PARAMS['num_heads'],
                mlp_size: int = HYPER_PARAMS['MLP_size'],
                dropout: float = 0.1):
        super().__init__()

        # Converts image into a sequence of patch embeddings
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        # Learnable [CLS] token that is prepended to the patch sequence
        self.cls_token = class_token

        # Learnable positional embedding added to patch + cls token embeddings
        self.pos_embed = position_embedding

        # Optional dropout after adding positional embeddings
        self.pos_dropout = nn.Dropout(dropout)

        # Stack of transformer encoder blocks
        self.encoder_layers = nn.Sequential(*[
            Encode(
                MSA=MSA,               # Multi-head self-attention block
                MLP=MLP,               # Feed-forward block
                hidden_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=dropout
            )
            for _ in range(depth)
        ])

        # Final normalization after all transformer blocks
        self.encoder_norm = nn.LayerNorm(embedding_dim)

        # Classification head using only the [CLS] token
        self.head = MLP_Head

        # Apply weight initialization as done in the official ViT implementation
        self._init_weights()

    def _init_weights(self):
        """
        Weight initialization to match the pretrained ViT-B/16 weights.
        - Position embeddings and class token initialized with truncated normal.
        - Linear layers are initialized with truncated normal, biases to 0.
        - LayerNorm weights to 1, biases to 0.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through the model:
        - Convert image to patches and embed them.
        - Add the class token.
        - Add positional encoding.
        - Pass through transformer encoder blocks.
        - Extract the [CLS] token for classification.
        """
        x = self.patch_embed(x)  # Shape: (B, N, D) where N = number of patches, D = embedding_dim

        # Expand class token to batch size and prepend to patch embeddings
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1 + N, D)

        # Add positional embeddings and apply dropout
        x = x + self.pos_embed                        # (B, 1 + N, D)
        x = self.pos_dropout(x)

        # Pass through transformer blocks
        x = self.encoder_layers(x)
        x = self.encoder_norm(x)

        # Classification using the [CLS] token (index 0)
        x = self.head(x[:, 0])                        # (B, num_classes)
        return x
