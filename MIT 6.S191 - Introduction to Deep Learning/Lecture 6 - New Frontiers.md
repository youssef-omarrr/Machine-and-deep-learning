# Lecture 6: New Frontiers
### [Video Link](https://www.youtube.com/watch?v=HLKo4fJx_7k&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=7&ab_channel=AlexanderAmini)
### [Slides Link](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L6.pdf)

## 1. Autoencoders: Denoising & Undercomplete

- **Autoencoder architecture**:
    
    - Encoder: compresses input x into latent code $h = f_\theta(x)$.
        
    - Decoder: reconstructs x from h: $\hat{x} = g_\phi(h)$.
        
    - Minimize reconstruction loss (e.g., MSE or binary cross-entropy).
        
- **Undercomplete AE**: latent dimension < input dimension → forces learning of compressed structure.
    
- **Denoising AE**: input is corrupted (e.g., with noise), but the network learns to reconstruct the clean version—learns robust features and manifold structure.
    

---

## 2. Variational Autoencoders (VAEs)

- **VAE** introduces a probabilistic latent space:
    
    - Encoder outputs mean & log-variance of a Gaussian distribution over latent code z.
        
    - Use reparameterization trick: z=μ+σ⊙ϵ, ϵ∼N(0,I).
        
    - Decoder maps z back to distribution over inputs.
        
- The loss = reconstruction loss + KL divergence with prior N(0,I):
    
     $$\mathcal{L}(x) = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x)\parallel p(z))$$

---

## 3. Other Unsupervised Models

- **t-SNE** & **UMAP**: non‑linear dimensionality reduction techniques (for visualization).
    
- **Contrastive learning** (e.g., SimCLR, MoCo): learn representations by pulling together augmentations of the same image and pushing apart different images.
    

---

## 🔍 PyTorch Examples

### A. Simple Autoencoder (for MNIST-like data)

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Training loop
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(10):
    epoch_loss = 0
    for x,_ in train_loader:
        x_hat = model(x)
        loss = criterion(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
```

---

### B. Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 400)
        self.fc_out = nn.Linear(400, 28*28)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = torch.relu(self.fc_dec(z))
        return torch.sigmoid(self.fc_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1,28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize, train with similar loop as above, using loss_fn.
```

---