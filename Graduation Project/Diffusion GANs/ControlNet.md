### Sources:
- [ControlNet with Diffusion Models | Explanation and PyTorch Implementation](https://www.youtube.com/watch?v=n6CwImm_WDI&t=15s)
---

# **ControlNet

## **1. What ControlNet Is**

ControlNet is a method for adding **extra control signals** (edge maps, segmentation masks, human poses, depth maps, etc.) to a **pre-trained diffusion model** (such as Stable Diffusion) **without breaking or forgetting** its original capabilities.

It allows you to guide the generation process with spatial information **while preserving the base model’s quality and diversity**.

---

## **2. The Core Problem**

Fine-tuning large diffusion models on a small conditional dataset leads to:

* **Overfitting** on the new condition.
* **Catastrophic forgetting** (the model loses its generative ability).
* Loss of **style diversity** and **image fidelity**.

ControlNet solves this by preserving the original model’s weights entirely.

---

## **3. Key Ideas Behind ControlNet**
![](../../Reinforcement%20Learning/imgs/PastedImage.png)
### **A. Freeze the Original UNet**

* The base diffusion model (e.g., Stable Diffusion’s UNet) is **completely frozen**.
* None of its weights change.
* This keeps the model’s prior knowledge intact.

### **B. Add a Trainable Copy of Specific Blocks**

* An **exact structural copy** of the UNet’s **encoder and middle blocks** is made.
* These copied blocks **are trainable**.
* The *decoder* is **not** copied, only the *encoder/mid* parts.

### **C. Zero Convolution Layers (Important)**

* Each copied block outputs into a **1×1 convolution initialized to zero**.
* At the start of training, these outputs are exactly zero.
* This ensures:
  * No harmful gradients reach the frozen model.
  * The model starts with *identical behavior* to the original UNet.
  * The influence of the control signal increases gradually during training.

### **D. Residual Injection**

* The output of each copied block (after zero-conv) is **added** to the output of the frozen block at the same level.
* This creates a **residual correction** that represents the influence of the control signal.

---

## **4. How ControlNet Integrates Into Stable Diffusion**
![](../../Reinforcement%20Learning/imgs/PastedImage-48.png)

![](../../Reinforcement%20Learning/imgs/PastedImage-49.png)
### **A. Stable Diffusion’s UNet Structure**

* **Down blocks (encoder)**
* **Mid block**
* **Up blocks (decoder)**

ControlNet modifies only the *encoder* and *mid blocks*, because these carry spatial features early in the network.

### **B. What Is the Hint / Conditioning Input?**

The control signal (called a **hint**) may be:

* Canny edges
* Segmentation masks
* Depth maps
* Human pose keypoints
* Normal maps
* Scribbles, sketches, etc.

This hint goes through a **Hint Encoder**:

* A stack of convolutions that converts the input into the same spatial size and channel depth as the first encoder block.
* Its output is added to the first layer of the trainable copy UNet.

### **C. Data Flow Summary**

1. Input image (noisy sample) enters both the frozen UNet and the trainable copy.
2. The control hint enters the Hint Encoder.
3. The Hint Encoder output is added to the input of the first copied block.
4. Each copied block produces feature maps → passes them through zero-conv.
5. Zero-conv outputs are injected into the corresponding frozen block as residuals.
6. The decoder remains completely frozen and processes the combined features.

### **D. Trainable Parameters**

Only the following update during training:

* The trainable UNet copy (encoder + middle blocks)
* The Hint Encoder (convs)
* The Zero-Convolution layers
  Everything else, including the original UNet and the text encoder, stays frozen.

---

## **5. Why ControlNet Works So Well**

* Uses the pre-trained model’s knowledge without modifying it.
* Adds small, precise residuals for spatial control.
* Zero-conv allows stable learning from scratch.
* The model can learn a new conditional mapping without forgetting old skills.
* No need for expensive full-model fine-tuning.

---

## **6. Example: MNIST Toy Experiment**

* A basic DDPM model is trained.
* ControlNet uses **Canny edges of digits** as the condition.
* With only **one epoch of training**, the model:

  * Generates digits that follow the provided edge maps.
  * Still maintains clean DDPM sampling behavior.

This demonstrates how quickly the model learns spatial control because the base model is frozen.

---

## **7. Additional Practical Notes**
![](../../Reinforcement%20Learning/imgs/PastedImage-50.png)
### **A. ControlNet Does Not Replace the Base Model**

It **adds** capability, it does not modify or weaken the original generator.

### **B. Multiple ControlNets Can Be Stacked**

You can combine:

* Canny + Depth
* Pose + Segmentation
* etc.

### **C. Training Is Fast**

Because:

* Only a small subnetwork is trained.
* The base model is reused.

### **D. Why Not Only Train a Small Branch?**

ControlNet copies whole blocks to maintain:

* Matching architecture with the frozen model
* Exact feature alignment at every level

---
