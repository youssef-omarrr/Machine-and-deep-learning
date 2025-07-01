from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps
import os

app = Flask(__name__)

# Model defination
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
    

# Create an instance of the model
model = MNIST_model(input_shape= 1,
                    hidden_units= 32,
                    output_shape= 10)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "MNIST_Model.pth")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# # Test evaluating the model
# from helper_functions import eval_model
# from torchmetrics import Accuracy
# from torchvision import datasets
# from torch.utils.data import DataLoader

# BATCH_SIZE = 32
# # Download and transform test data
# test_data = datasets.MNIST(
#     root="data",
#     train= False,
#     transform= transforms.ToTensor(),
#     download= True
# )
# test_dataloader = DataLoader(test_data,
#                             batch_size = BATCH_SIZE,
#                             shuffle = False,
# )

# acc_metric = Accuracy(task="multiclass", num_classes=10)
# loss_fn = nn.CrossEntropyLoss()
# optm = torch.optim.SGD(params= model.parameters(),
#                     lr = 0.1)

# number_model_results = eval_model(model=model, 
#                             data_loader=test_dataloader,
#                             loss_fn=loss_fn, 
#                             accuracy_fn=acc_metric
# )

# print(number_model_results)

# Define pre-processing
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def Preprocessing_input(pil_img):
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


# Serve the drawing page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error' : 'No image file provided'}), 400
    
    file = request.files['file']
    
    img = Image.open(file.stream)
    img_tensor = Preprocessing_input(img)
    
    if img_tensor is None:
        return jsonify({'prediction': 'blank input'})
    
    with torch.inference_mode():
        outputs = model(img_tensor)
        _, prediction = torch.max(outputs.data, 1)
        
    return jsonify({"Predicted" : int(prediction)})
        
        
        
if __name__ == '__main__':
    app.run(debug = True)