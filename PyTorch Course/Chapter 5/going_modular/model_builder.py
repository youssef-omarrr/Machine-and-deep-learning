"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""

import torch
from torch import nn

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
        input_shape (int): Number of input channels (e.g., 3 for RGB images).
        hidden_units (int): Number of output channels (filters) for convolutional layers.
        output_shape (int): Number of output classes for classification.
    """
    def __init__(self, input_shape: int,
                hidden_units: int,
                output_shape: int):
        super().__init__()

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            # First convolution layer
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1  # preserve spatial size
            ),
            nn.ReLU(),

            # Second convolution layer
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            # Downsample feature map (halves height and width)
            nn.MaxPool2d(kernel_size=2)
        )

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        # Automatically calculate the number of features going into the linear layer
        with torch.no_grad():
            # Simulate a dummy input tensor to pass through the conv layers
            temp = torch.zeros(1, input_shape, 64, 64)  # batch size 1, 64x64 image
            dummy = self.conv_block2(self.conv_block1(temp))
            num_features = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]  # flatten dims

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten 3D feature map to 1D vector
            nn.Linear(
                in_features=num_features,
                out_features=output_shape  # One output per class
            )
        )

    def forward(self, x):
        """Defines the forward pass of the network."""
        # Pass input through both convolutional blocks and the classifier
        return self.classifier(self.conv_block2(self.conv_block1(x)))
