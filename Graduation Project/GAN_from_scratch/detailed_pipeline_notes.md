# Training Metrics Summary

## Diff_loss (Diffusion Loss):
Measures how well the model learns to remove noise and reconstruct clean images.
> Lower is better.

## gen_adv_loss (Generator Adversarial Loss):
Measures how well the generator fools the discriminator into thinking fake images are real.
> Lower is better.

## Disc_loss (Discriminator Loss):
Measures how well the discriminator separates real images from fake ones.
> - Medium values are healthy. 
> - Too low = discriminator too strong.
> - Too high = discriminator too weak.

---

# Pipeline Explanation

---

## Overall Goal
The model learns to create **fake medical images** that look real. It combines two powerful techniques: **Diffusion** (gradual image creation) and **GAN** (real vs fake discrimination).

---

## 1. Diffusion Scheduler

**What it does:** Controls the noise addition process

Think of it like **adding fog to a clear photo**:
- Starts with a clear image (your real medical scan)
- Gradually adds more and more noise over 1000 steps
- By step 1000, the image is pure random noise

**Why?** The model learns to **reverse this process** - turning noise back into clear images!

```
Clear Image → Add noise → Add more noise → ... → Pure Noise
   Step 0         Step 1       Step 500           Step 1000
```

### Code Implementation:

```python
class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T = timesteps  # Total number of noise steps
        
        # Beta: Controls how much noise to add at each step
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Alpha: The complement of beta (1 - beta)
        self.alphas = 1.0 - self.betas
        
        # Alpha_hat: Cumulative product - total noise accumulated up to step t
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
```

**Key variables:**
- `betas`: How much noise to add at each step (increases over time)
- `alphas`: How much signal to keep at each step
- `alpha_hat`: Total signal remaining after t steps

```python
def add_noise(self, x0, t):
    """
    Add noise to clean image x0 at timestep t
    
    Args:
        x0: Clean image
        t: Timestep (0 to 1000)
    
    Returns:
        x_t: Noisy image at timestep t
        noise: The noise that was added
    """
    noise = torch.randn_like(x0)  # Random noise, same shape as image
    a_hat = self.alpha_hat[t].view(-1,1,1,1)  # Get alpha_hat for timestep t
    
    # Formula: x_t = sqrt(alpha_hat) * x0 + sqrt(1 - alpha_hat) * noise
    # This is the mathematical way to add exactly the right amount of noise
    x_t = torch.sqrt(a_hat)*x0 + torch.sqrt(1-a_hat)*noise
    return x_t, noise
```

---

## 2. UNet Generator (G)

**What it does:** The artist that creates images

This is the **brain** of your system. It's a special neural network that:
- Takes **noisy images** as input
- Predicts what **noise to remove**
- Has a U-shape architecture:
  - **Down path**: Compresses image, captures big features
  - **Bottleneck**: Processes deep information
  - **Up path**: Rebuilds image, adds details back

**Think of it like:** A photo restoration expert who removes scratches (noise) from old photos.

### Building Blocks:

```python
class DoubleConv(nn.Module):
    """
    Two convolution layers with normalization and activation
    This is the basic building block of the UNet
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # First convolution
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # Extract features
            nn.GroupNorm(8, out_ch),                  # Normalize (helps training)
            nn.ReLU(),                                # Activation (non-linearity)
            
            # Second convolution
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)
```

### Full UNet Architecture:

```python
class UNet(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        
        # ENCODER (Downsampling path) - Captures context
        self.down1 = DoubleConv(in_ch, base)      # 128x128 → features
        self.down2 = DoubleConv(base, base*2)     # 64x64 → more features
        self.down3 = DoubleConv(base*2, base*4)   # 32x32 → even more features
        
        self.pool = nn.MaxPool2d(2)  # Reduces size by half
        
        # BOTTLENECK - Deepest processing
        self.mid = DoubleConv(base*4, base*8)     # 16x16 → deepest features
        
        # DECODER (Upsampling path) - Rebuilds image with details
        self.up3 = DoubleConv(base*8 + base*4, base*4)  # 32x32
        self.up2 = DoubleConv(base*4 + base*2, base*2)  # 64x64
        self.up1 = DoubleConv(base*2 + base, base)      # 128x128
        
        # Final layer - output predicted noise
        self.final = nn.Conv2d(base, in_ch, 1)
```

```python
def forward(self, x):
    """
    Forward pass through UNet
    
    The U-shape:
    - Go down (encoder): Extract features at multiple scales
    - Bottleneck: Process deepest features
    - Go up (decoder): Rebuild image, concatenating encoder features (skip connections)
    """
    
    # ENCODER - going down
    d1 = self.down1(x)           # First level features
    d2 = self.down2(self.pool(d1))  # Second level (smaller, more abstract)
    d3 = self.down3(self.pool(d2))  # Third level
    
    # BOTTLENECK
    mid = self.mid(self.pool(d3))   # Deepest level
    
    # DECODER - going up
    # Note: We concatenate encoder features (skip connections)
    # This preserves fine details from early layers
    
    u3 = F.interpolate(mid, scale_factor=2)      # Upsample
    u3 = self.up3(torch.cat([u3, d3], dim=1))    # Concatenate + process
    
    u2 = F.interpolate(u3, scale_factor=2)
    u2 = self.up2(torch.cat([u2, d2], dim=1))
    
    u1 = F.interpolate(u2, scale_factor=2)
    u1 = self.up1(torch.cat([u1, d1], dim=1))
    
    return self.final(u1)  # Output: predicted noise
```

**Why skip connections?** Concatenating encoder features helps preserve fine details that might be lost during downsampling.

---

## 3. Discriminator (D)

**What it does:** The quality inspector

This is the **critic** that tells real from fake:
- Looks at images
- Outputs: "This looks REAL" or "This looks FAKE"
- Forces the Generator to improve

**Think of it like:** An art expert who can spot fake paintings.

### Code Implementation:

```python
class Discriminator(nn.Module):
    """
    Judges whether an image is real or fake
    Architecture: Series of convolutional layers that downsample the image
    """
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: 128x128 → 64x64
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),  # LeakyReLU prevents "dying ReLU" problem
            
            # Layer 2: 64x64 → 32x32
            nn.Conv2d(base, base*2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base*2),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 32x32 → 16x16
            nn.Conv2d(base*2, base*4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base*4),
            nn.LeakyReLU(0.2),
            
            # Final layer: 16x16 → 1x1 (single score)
            nn.Conv2d(base*4, 1, 4, padding=0)
        )
    
    def forward(self, x):
        """
        Returns a score (not probability!)
        Higher score = more likely to be real
        Lower score = more likely to be fake
        """
        return self.model(x)
```

---

## 4. Loss Functions

### A) Diffusion Loss:

```python
def diffusion_loss(model, x0, t, scheduler):
    """
    Trains the model to predict and remove noise
    
    Args:
        model: UNet generator
        x0: Clean real image
        t: Random timestep
        scheduler: Diffusion scheduler
    
    Returns:
        loss: How well the model predicted the noise
    """
    # Add noise to clean image
    x_t, noise = scheduler.add_noise(x0, t)
    
    # Ask model to predict what noise was added
    pred_noise = model(x_t)
    
    # Compare predicted noise to actual noise
    # Lower loss = better at predicting noise
    return F.mse_loss(pred_noise, noise)
```

### B) GAN Losses:

```python
def gan_losses(D, real, fake):
    """
    Trains the discriminator to distinguish real from fake
    
    Args:
        D: Discriminator
        real: Real medical images
        fake: Generated fake images
    
    Returns:
        real_loss: How well D identifies real images
        fake_loss: How well D identifies fake images
    """
    # Discriminator should output HIGH scores for real images
    # softplus(-D(real)) penalizes when D outputs low scores for real
    real_loss = F.softplus(-D(real)).mean()
    
    # Discriminator should output LOW scores for fake images
    # softplus(D(fake)) penalizes when D outputs high scores for fake
    fake_loss = F.softplus(D(fake)).mean()
    
    return real_loss, fake_loss
```

---

## 5. Training Process

```python
def train(dataloader, image_channels=1, timesteps=1000, device="cuda", 
          epochs=100, save_dir="checkpoints"):
    """
    Main training loop combining Diffusion and GAN training
    """
    
    # Create save directory for checkpoints
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize models
    G = UNet(in_ch=image_channels).to(device)           # Generator
    D = Discriminator(in_ch=image_channels).to(device)  # Discriminator
    scheduler = DiffusionScheduler(timesteps, device=device)
    
    # Optimizers (Adam is standard for GANs and Diffusion)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
    
    for epoch in range(epochs):
        pbar = tqdm.tqdm(dataloader)
        
        for batch in pbar:
            # Handle different dataloader formats (Albumentations returns dict)
            if isinstance(batch, dict):
                real = batch['image']
            elif isinstance(batch, (list, tuple)):
                real = batch[0]
            else:
                real = batch
            
            real = real.to(device)
            
            # ===== STEP 1: DIFFUSION TRAINING =====
            # Sample random timesteps for each image in batch
            t = torch.randint(0, timesteps, (real.size(0),)).to(device)
            
            # Calculate diffusion loss (noise prediction)
            diff_loss = diffusion_loss(G, real, t, scheduler)
            
            # ===== STEP 2: GAN TRAINING =====
            
            # Generate fake images by denoising
            # Start with noisy version of real image (t=0, minimal noise)
            noisy, _ = scheduler.add_noise(real, torch.zeros_like(t))
            fake = G(noisy.detach())
            
            # --- Train Discriminator ---
            # D should say "real" for real images, "fake" for fake images
            D_real, D_fake = gan_losses(D, real, fake.detach())
            D_loss = D_real + D_fake
            
            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()
            
            # --- Train Generator to fool Discriminator ---
            # G wants D to output HIGH scores for fake images
            adv_loss = F.softplus(-D(fake)).mean()
            
            # ===== STEP 3: COMBINED GENERATOR LOSS =====
            # 90% diffusion loss + 10% adversarial loss
            # Diffusion ensures proper denoising, GAN ensures realism
            G_loss = diff_loss + 0.1 * adv_loss
            
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()
            
            # Update progress bar
            pbar.set_description(
                f"Epoch {epoch} | Diff {diff_loss:.4f} | Adv {adv_loss:.4f} | D {D_loss:.4f}"
            )
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(G, D, opt_G, opt_D, epoch, save_dir)
    
    # Save final model
    save_checkpoint(G, D, opt_G, opt_D, epochs-1, save_dir, final=True)
    
    return G, D, scheduler
```

**Training Summary:**
```
For each batch:
1. Calculate diffusion loss (how well G removes noise)
2. Train D to distinguish real vs fake
3. Train G to fool D
4. Combine losses and update G
5. Repeat
```

---

## 6. Generating New Images (Sampling)

```python
def sample(G, scheduler, n=8, image_size=128, device="cuda"):
    """
    Generate new synthetic medical images from random noise
    
    This is the REVERSE diffusion process:
    Start with noise → gradually denoise → final clean image
    
    Args:
        G: Trained generator
        scheduler: Diffusion scheduler
        n: Number of images to generate
        image_size: Size of output images
    
    Returns:
        Generated images
    """
    G.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    with torch.inference_mode():  # Faster inference, no gradient calculation
        # Start with pure random noise
        x = torch.randn(n, 1, image_size, image_size).to(device)
        
        # Reverse diffusion: Go from t=999 down to t=0
        for t in reversed(range(scheduler.T)):
            # Get noise schedule parameters for this timestep
            a_hat = scheduler.alpha_hat[t]  # Cumulative signal
            a = scheduler.alphas[t]          # Signal at this step
            b = scheduler.betas[t]           # Noise at this step
            
            # Ask generator: "What noise is in this image?"
            noise_pred = G(x)
            
            # Add small random noise (except at final step)
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # DDPM sampling formula:
            # Remove predicted noise, add small random noise for next step
            x = (1/torch.sqrt(a)) * (x - ((1-a)/(torch.sqrt(1-a_hat))) * noise_pred) + torch.sqrt(b) * noise
    
    G.train()  # Set back to training mode
    return x
```

**Step-by-step visualization:**
```
t=999: [Pure noise]
t=800: [Vague shapes emerging]
t=500: [Basic structure visible]
t=200: [Details forming]
t=0:   [Clear synthetic medical image]
```

---

## Why This Combination Works

**Diffusion alone:**
- Great at learning image structure
- Stable training
- Can be slow and blurry

**GAN alone:**
- Creates sharp, realistic images
- Fast sampling
- Unstable training, mode collapse

**Together:**
- Diffusion provides: Stability, good structure, diverse outputs
- GAN provides: Sharpness, realism, speed
- Result: High-quality, diverse, realistic synthetic medical images

---

## Complete Usage Example

```python
# ===== TRAINING =====
G, D, scheduler = train(
    dataloader, 
    image_channels=1, 
    epochs=200, 
    save_dir="checkpoints"
)

# ===== LOAD SAVED MODEL (skip training) =====
G, D, scheduler = load_checkpoint("checkpoints/final_model.pth")

# ===== GENERATE SYNTHETIC DATA =====
synthetic_images = generate_and_visualize(
    G, scheduler,
    n=16,                               # Generate 16 images
    image_size=128,
    save_grid="results/grid.png",       # Save grid
    save_individual="results/samples"   # Save each image
)
```