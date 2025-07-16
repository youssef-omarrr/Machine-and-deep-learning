# Lecture 4: Deep Generative Modeling 

### [Video Link](https://www.youtube.com/watch?v=SdTZAMDKrNY&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4&ab_channel=AlexanderAmini)
### [Slides Link](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L4.pdf)

# NOTES:
## Supervised vs. Unsupervised Learning

![Alt text](imgs/Pasted%20image%2020250706211603.png)
### Supervised Learning:

- Goal: Learn a mapping from inputs x to outputs y using labeled data.
    
- Training data: $(x, y)$ pairs.
    
- Objective: Minimize prediction error on known labels.
    
- Example: Image classification, sentiment analysis.
    

### Unsupervised Learning:

- Goal: Discover hidden structure in **unlabeled** data.
    
- No $y$ values given.
    
- Example: Clustering, dimensionality reduction, **generative modeling**.
    

---
### **Standard (Discriminative) Learning:**

- Focus: **Learn a function or mapping**
    
    - Example: f(x)→y
        
- Works with **tensors directly** (fixed input-output pairs).
    
- Example:
    
    - Image → Label
        
    - Input vector → Regression value
        
- Objective: Minimize prediction error (cross-entropy, MSE, etc.)
---
## Generative Modeling

![Alt text](imgs/Pasted%20image%2020250706211640.png)
### What is a Generative Model?

- Learns the **underlying data distribution** $p(x)$.

- Can **generate new samples** similar to the training data.

- Can be **explicit** (models $p(x)$ directly) or **implicit** (learns to generate samples without an explicit form of $p(x)$.

-  Focus: **Learn the data distribution** p(x) or p(x∣z)
- Works with **probability distributions**, often using **sampling and latent variables**.
- Goal: Generate new samples that “look like” the training data.

### Uses of Generative Models:
- Detecting something new or rare.

- Data generation
    
- Data imputation
    
- Semi-supervised learning
    
- Denoising
    
- Compression
    
---
### In short:
| Type                 | Learns...                                          | Works with...             |
| -------------------- | -------------------------------------------------- | ------------------------- |
| Discriminative Model | A direct function \( $f(x) \rightarrow y$ \)       | Tensors and mappings      |
| Generative Model     | A distribution \( $p(x)$ \) or \( $p(x \mid z)$ \) | Probability distributions |

---
## Latent Variable Models

![Alt text](imgs/Pasted%20image%2020250706212225.png)

### Motivation:

A **latent variable** is a variable that is **not directly observed** but is **inferred** from observed data. It represents **hidden or underlying factors** that help explain the structure or relationships in the observed variables.

In deep learning, latent variables are often used to capture high-level abstract features (e.g., object identity, style) and are typically denoted as z in models like **VAEs** or **probabilistic graphical models**.

- Introduce **latent variables** z to explain observed data x.
    
- Model joint distribution:
    $$p(x, z) = p(x \mid z) p(z)$$

- Marginal likelihood:
    $$p(x) = \int p(x \mid z) p(z) \, dz$$


### Benefits:

- Models high-dimensional data using **low-dimensional latent codes** z.
    
- Encodes structure, semantics, and variations.

## Autoencoders: background

An **autoencoder** is a type of neural network designed to **learn efficient representations** (encodings) of input data, typically for **dimensionality reduction**, **denoising**, or **feature learning**.

It consists of two main parts:

1. **Encoder**:  
    Maps the input x to a lower-dimensional **latent representation** z.
2. **Decoder**:  
    Reconstructs the input from the latent representation.

The network is trained to minimize the **reconstruction error** between xx and x^\hat{x}, often using a loss like:
$$\mathcal{L}(x, \hat{x}) = \|x - \hat{x}\|^2$$

### Background & Motivation

- Inspired by **principal component analysis (PCA)** but with non-linear functions.
    
- Autoencoders learn to **compress** information into a **latent space**, then reconstruct it.
    
- Foundation for more advanced models like **Variational Autoencoders (VAEs)** and **representation learning** techniques.

![Alt text](imgs/Pasted%20image%2020250706212457.png)

---
## Variational Autoencoders (VAEs)

![Alt text](imgs/Pasted%20image%2020250706212844.png)
### Overview:

- VAEs are **latent variable generative models**.
    
- Combine ideas from:
    
    - Probabilistic graphical models
        
    - Autoencoders
        
    - Variational inference
        

---

### VAE Architecture:

![Alt text](imgs/Pasted%20image%2020250706212931.png)

1. **Encoder (Inference Network):** we want to compute the distribution of z given input x
    
   $$q_\phi(z \mid x)$$
    
    Approximates the posterior $p(z \mid x)$, with $\phi$ as weighting.
    
2. **Decoder (Generative Network):** we want to re-compute the distribution of input z given z
    
    
    $$p_\theta(x \mid z)$$

    
    Reconstructs input from latent variable z, with $\theta$ as weighting.


---

### Objective: Variational Lower Bound (ELBO)

We cannot compute the exact marginal likelihood $log p(x)$, so we maximize a lower bound:


$$\log p(x) \geq \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x) \| p(z))$$

- **First term**: Reconstruction accuracy
    
- **Second term**: Regularization (encourages q to match prior)
    

---

### Reparameterization Trick

### Problem: 
In Variational Autoencoders (VAEs), we **sample** the latent variable $z∼N(μ,σ2).$  
However, **sampling is a stochastic operation** (**non-differentiable operation** — meaning gradients can't flow through it, and training with backpropagation breaks.), and **we cannot backpropagate through a stochastic node**, because gradient-based optimization requires **deterministic** operations.

![Alt text](imgs/Pasted%20image%2020250706213742.png)

### Solution: 
We reparameterize the sampling step to make z a **deterministic function of μ, σ, and a random noise ϵ**:
$$ z = \mu + \sigma \cdot \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I) $$
- μ,σ : outputs of the encoder (learned deterministically)
- ϵ: random noise sampled independently

This **moves the stochasticity to the input** (through ϵ), allowing gradients to flow through μ and σ during backpropagation.

![Alt text](imgs/Pasted%20image%2020250706213701.png)![Alt text](imgs/Pasted%20image%2020250706213805.png)

---

### VAE Summary

- Encodes inputs into a **probabilistic latent space**.
    
- Learns to **generate** and **reconstruct** data (with no labels).
    
- Uses **KL divergence** to regularize the latent space.
    
- Limitations:
    
    - Tends to produce blurry images.
        
    - Limited expressiveness in some cases.
        
	![Alt text](imgs/Pasted%20image%2020250706214255.png)
	
**In simpler terms:** **Reparameterizing** means **rewriting the random sampling** in a way that makes it compatible with backpropagation by **pushing randomness to the input** and keeping the rest deterministic.

---
## Generative Adversarial Networks (GANs)

### Overview:

- GANs are **implicit generative models**.
    
- Use two networks trained in **adversarial** fashion:
    
    - **Generator** G(z): maps noise $z∼N(0,I)$ to data space.
        
    - **Discriminator** D(x): distinguishes between real and generated samples.
	![Alt text](imgs/Pasted%20image%2020250706214943.png)
---

### GAN Objective

Min/max game:


$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$


- D: wants to maximize probability of classifying real vs. fake.
    
- G: wants to fool D into thinking generated data is real.
    

---

### GAN Training Steps

1. Train D to classify real vs. generated.
    
2. Train G to generate more realistic samples.
    
3. Repeat until equilibrium is reached.
    ![Description](imgs/Pasted%20image%2020250706215108.png)


---

### Problems with GANs

- **Mode collapse**: Generator produces limited variety.
    
- **Training instability**: Difficult optimization dynamics.
    
- **No explicit density**: Cannot evaluate likelihoods.
    

---

### GAN Variants

- **DCGAN**: Deep convolutional GAN
    
- **Conditional GAN (cGAN)**:
	![Image](imgs/Pasted%20image%2020250706215253.png)


    $G(z, y), \quad D(x, y)$
    
    Adds class labels y as conditioning inputs.
    
- **CycleGAN**: Image-to-image translation without paired data.
	![Image](imgs/Pasted%20image%2020250706215323.png)

    
- **Wasserstein GAN**: Improves stability using Earth Mover’s distance.
    

---

## VAEs vs. GANs

|Feature|VAE|GAN|
|---|---|---|
|Output quality|Blurry|Sharp|
|Inference model|Yes (encoder)|No|
|Latent space structure|Structured, interpretable|Less structured|
|Training stability|Stable|Can be unstable|
|Likelihood|Approximate, via ELBO|Not available|
|Use cases|Reconstruction, representation|High-quality generation|

---

## Summary

- Generative models learn p(x) to generate new data.
    
- Latent variable models introduce hidden z to structure learning.
    
- VAEs use variational inference to approximate posteriors and reconstruct data.
    
- GANs use adversarial loss to generate highly realistic data.
    
- Tradeoffs exist between **likelihood-based methods** (VAEs) and **sample-quality-focused methods** (GANs).
    
