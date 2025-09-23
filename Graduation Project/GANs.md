# Types of GANs and Related Models for Tabular Data

## 🔹 GAN-based Models

### 1. **CTGAN (Conditional Tabular GAN, part of the SDV project)**
- Designed specifically for **tabular data** with both continuous and categorical features.
- Uses **conditional sampling** to handle imbalanced categories.
- Often the **best starting point** for most use cases.
- Stronger with **large datasets** that have complicated, nonlinear relationships.
- ⚠️ Can suffer from **mode collapse** and struggles with **rare categories**.
- Implemented in `sdv.single_table.CTGANSynthesizer` & `ctgan.CTGAN`
---

### 2. **CTAB-GAN / CTAB-GAN+ (Conditional Tabular GAN with Attention and Balancing)**
- An **enhanced version of CTGAN** that adds:
  - **Attention mechanisms** for better feature interaction modeling.
  - **Normalization strategies** for improved stability.
- Consistently **outperforms plain CTGAN** on real-world tabular datasets, especially with many categorical fields.
- Often considered **best-in-class for tabular GANs**.
- No implemntation in `SDV`, but this repo implemented it [Team-TUB/CTAB-GAN-Plus](https://github.com/Team-TUD/CTAB-GAN-Plus)

---

### 3. **CopulaGAN**
- A **hybrid model**:
  - First applies a **copula transformation** to normalize feature distributions.
  - Then uses a **GAN** to capture complex dependencies.
- Can work better than CTGAN when the dataset has a **mix of numeric and categorical features with correlations**.
- ⚠️ Still less expressive than CTAB-GAN+.

---

### 4. **Custom GANs (e.g., PyTorch implementations / WGAN-GP)**  
- **WGAN-GP = Wasserstein GAN with Gradient Penalty.**
- Useful if you want **full control** over:
  - Network architecture.
  - Loss functions.
  - Training strategies.
- **Best suited for numeric-heavy datasets**, where categorical encoding is simpler.
- WGAN-GP improves stability and mitigates mode collapse.
- ⚠️ More engineering effort required for handling categorical features.

---

### 5. **Diffusion Models (for Tabular Data)**
- A newer direction for synthetic tabular data generation.
- Can produce **higher-fidelity data** with better **mode coverage** than GANs.
- ⚠️ Training is **computationally heavier** and more complex.
- Worth considering for **state-of-the-art results** or research purposes.

---

## 🔹 VAE-based Models

### 6. **TVAE (Tabular Variational Autoencoder)**
- A VAE (Variational Autoencoder) adapted for tabular data.
- Learns a **latent representation** of the dataset, then decodes it back to synthetic rows.
- Compared to CTGAN:
  - More **stable training**.
  - Often better for **continuous columns**.
- Compared to Copula models:
  - Captures **nonlinear dependencies**.
  - Requires **more data**.
- Great balance between CTGAN’s power and Copula’s simplicity.

---

## 🔹 Copula-based Models

### 7. **Gaussian Copula Synthesizer**
- Works in **two steps**:
  1. Learn the **distribution of each column** separately.
  2. Learn the **relationships between columns** via a copula.
- Process:
  - Transform columns into a common **normal scale**.
  - Learn dependencies in that space.
  - Transform back to generate synthetic rows.

#### ✅ Strengths
- Very **stable** and **fast**.
- Works well for **small to medium datasets**.
- No mode collapse issues.

#### ⚠️ Weaknesses
- Limited in capturing **complex nonlinear relationships**.
- Struggles with **high-cardinality categorical features**.

---

## ⚖️ Rule of Thumb

- Start with **CTGAN** → balanced, easy, effective.
- Use **CTAB-GAN+** for real-world datasets with many categorical fields.
- Try **TVAE** if you prefer stability and speed, especially for continuous-heavy datasets.
- Consider **Custom GANs** (e.g., WGAN-GP) if you need research flexibility or want to experiment with architectures.
- Watch **Diffusion models** if aiming for cutting-edge performance.
- Use **Gaussian Copula** if you need something **fast, stable, and simple** for smaller datasets.

---

## 📊 Quick Comparison Table

| Model                     | Full Name / Meaning                              | Pros ✅ | Cons ⚠️ | Best Use Case |
|----------------------------|--------------------------------------------------|--------|---------|---------------|
| **CTGAN**                 | Conditional Tabular GAN                          | Handles categorical + continuous, good first choice | Can mode-collapse, weak on rare categories | Large, mixed datasets |
| **CTAB-GAN / CTAB-GAN+**  | Conditional Tabular GAN with Attention & Balancing | Attention + normalization, best for tabular | Heavier to train than CTGAN | Real-world data with many categorical fields |
| **CopulaGAN**             | GAN combined with copula transformation          | Hybrid approach, balances copula + GAN | Less expressive than CTAB-GAN+ | Mixed data with correlations |
| **Custom GANs / WGAN-GP** | Wasserstein GAN with Gradient Penalty (custom architectures) | Full flexibility, stable with WGAN-GP | Requires custom engineering | Numeric-heavy datasets, research |
| **Diffusion Models**      | Probabilistic denoising diffusion for tabular data | State-of-the-art fidelity, great coverage | Computationally heavy, complex training | Cutting-edge experiments |
| **TVAE**                  | Tabular Variational Autoencoder                  | Stable, good for continuous data | Needs more data, weaker on small datasets | Balanced datasets with continuous focus |
| **Gaussian Copula**       | Copula-based statistical model                   | Very fast + stable, no mode collapse | Struggles with nonlinear deps & high-cardinality cats | Small to medium datasets, quick baseline |

---
