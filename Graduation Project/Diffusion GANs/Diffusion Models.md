# Diffusion Models

## Sources:
- [ControlNet with Diffusion Models | Explanation and PyTorch Implementation](https://www.youtube.com/watch?v=n6CwImm_WDI&t=15s)

---

Diffusion Models are a class of **generative models** that learn to create new data (like images or audio) by simulating a physical process of diffusing and then reversing it. They currently produce some of the most realistic synthetic images.

---

## Core Idea: The Two Processes

The model works in two main stages: a fixed **Forward Process** that adds noise, and a learned **Reverse Process** that removes it.

### 1. Forward Diffusion Process (Noising)
This process is fixed and requires **no training**. It gradually adds **Gaussian noise** to a clean data sample (e.g., an image $x_0$) over a series of steps (T) until the original data is completely transformed into pure random noise ($x_T$).

* **Clean Image** $\rightarrow$ **Slightly Noisy** $\rightarrow$ **Very Noisy** $\rightarrow$ **Pure Noise**
* The math ensures that the final result, $x_T$, is a simple, known **Gaussian distribution** (pure random noise).



### 2. Reverse Diffusion Process (Denoising/Generation)
This is the process the model **learns**. The goal is to start from pure random noise ($x_T$) and gradually reverse the noising steps to produce a clean, generated data sample ($x_0$).

* The model (often a **U-Net neural network**) is trained to predict and remove the small amount of noise added at each step in the reverse direction.
* **Pure Noise** $\rightarrow$ **Slightly Structured** $\rightarrow$ **Mostly Clean** $\rightarrow$ **New Image**

During training, the model is shown a noisy image $x_t$ and the known noise that was added to create it. The model learns a function to predict this noise $\epsilon_\theta(x_t, t)$, where $\theta$ are the model's parameters. By minimizing the difference between the predicted noise and the actual noise, the model learns the complex path back to a clean image.

---

## Key Models and Architectures

### Denoising Diffusion Probabilistic Models (DDPM)
**DDPM** is the foundational framework for modern diffusion models.

* **Mechanism:** It simplifies the training objective, focusing the model purely on predicting the noise ($\epsilon$) added at a given step $t$.
* **Training:** The model is trained to minimize the difference between the predicted noise $\epsilon_\theta(x_t, t)$ and the true noise $\epsilon$.
    $$\min_{\theta} || \epsilon - \epsilon_{\theta}(x_t, t) ||^2$$
* **Impact:** This simple, yet powerful, training objective led to unprecedented quality in generative image modeling.

### Stable Diffusion
Stable Diffusion is a popular, high-performing text-to-image diffusion model.

* **Key Idea: Latent Space:** Instead of running the diffusion process on the high-resolution pixel space of the image, Stable Diffusion works in a smaller **Latent Space** (a compressed, lower-dimensional representation of the image). This makes the process dramatically **faster** and requires less computing power.
* **Conditioning:** It uses a **text encoder** (like CLIP) to translate a text prompt (e.g., "A dog wearing a hat") into a numerical **embedding**. This embedding is injected into the U-Net via **cross-attention** layers, guiding the denoising process to match the described text.



### ControlNet
**ControlNet** is an extension that gives Stable Diffusion **extra control** over the spatial structure or pose of the generated image.

* **Mechanism:** It takes the weights of a pre-trained Stable Diffusion model and creates two copies:
    1.  A **locked copy** (frozen weights) that preserves the core generation capability.
    2.  A **trainable copy** that is fine-tuned on a specific conditional input (like a Canny edge map, depth map, or human pose skeleton).
* **Function:** The trainable copy learns the desired structural control (e.g., "keep this specific pose") and passes this information to the locked copy via **skip connections**. This allows for the generation of a new image that follows both the text prompt **and** the exact structural input from a reference image.
* **Example Uses:** *Canny Edges* (maintaining outlines), *OpenPose* (replicating a human pose), *Depth Map* (preserving 3D structure).

ControlNet allows researchers to fine-tune Stable Diffusion on new tasks **without sacrificing** its vast knowledge of image generation.

---