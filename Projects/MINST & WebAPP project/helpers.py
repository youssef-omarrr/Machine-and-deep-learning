import torch
from torch import nn
import os
from PIL import Image, ImageOps

def build_model(input_shape= 1,
                hidden_units= 32,
                output_shape= 10):
    
    class MNIST_model (nn.Module):
        def __init__(self, 
                    input_shape:int,
                    hidden_units:int,
                    output_shape:int):
            super().__init__()
            
            # First convolutional block
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                        out_channels= hidden_units,
                        kernel_size= 3),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2)
            )
            
            # Second convolutional block
            self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels= hidden_units,
                        kernel_size= 3),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2)
            )
            
            # Calculate the flattened feature size after conv blocks
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_shape, 28, 28)
                x = self.block_1(dummy_input)
                x = self.block_2(x)
                num_features = x.shape[1] * x.shape[2] * x.shape[3]
                
            # Classifier (fully connected layer)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features= num_features,
                    out_features= output_shape
                )
            )
            
        def forward (self, x:torch.Tensor):
            # Forward pass through conv blocks and classifier
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.classifier(x)
            
            return x
        
    return MNIST_model(input_shape,
                    hidden_units,
                    output_shape)

def Preprocessing_input(pil_img, transform):
    # Ensure 'static' directory exists
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
    # Convert to grayscale
    img = pil_img.convert('L')
    
    # Invert colors: black digit on white background
    img = ImageOps.invert(img)
    
    # Resize image to 28x28
    img = img.resize((28, 28))
    
    # Save as preview
    img.save(os.path.join("static", "received_input.png"))
    print("Image saved at:", os.path.abspath("static\\received_input.png"))
    
    tensor_img = transform(img).unsqueeze(0)  # shape: [1, 1, 28, 28]

    return tensor_img


# Function to train the model briefly using the corrected image and label
def REtrain_model(model, optimizer, transform, correct_label, img):
    MODEL_PATH = "MNIST_Model.pth"

    # === Load previous weights if they exist ===
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()  # Optional: set to eval before training
        print("Loaded existing model weights.")
        
    else:
        print("No previous weights found. Training from scratch.")
        
    # Initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    # Preprocess the image (convert to tensor and reshape)
    img_tensor = Preprocessing_input(img, transform)  # Expected shape: [1, 1, 28, 28]

    # Convert the correct label to a tensor
    correct_label = torch.tensor([int(correct_label)])

    # Set model to training mode
    model.train()

    # Forward pass through the model
    y_pred = model(img_tensor)  # Output logits for each class

    # Calculate the loss between prediction and true label
    loss = loss_fn(y_pred, correct_label)

    # Zero gradients from previous step
    optimizer.zero_grad()

    # Backpropagation: compute gradients
    loss.backward()

    # Perform one optimization step to update model weights
    optimizer.step()

    # Save the updated model state to disk
    torch.save(obj=model.state_dict(), f=MODEL_PATH)
    print(f"Model UPDATED")