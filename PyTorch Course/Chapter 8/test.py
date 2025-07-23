import torch
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, TensorDataset
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_b_16(weights=None).to(device)

    dummy_x = torch.randn(64, 3, 224, 224)
    dummy_y = torch.randint(0, 1000, (64,))
    ds = TensorDataset(dummy_x, dummy_y)

    loader = DataLoader(ds, batch_size=16, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()
    model.train()
    for epoch in range(1):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    print("Time taken:", time.time() - start)

if __name__ == "__main__":
    main()
