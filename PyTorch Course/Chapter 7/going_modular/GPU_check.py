import torch

def GPU_check():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "None")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    
    return 'cuda' if torch.cuda.is_available() else 'cpu'