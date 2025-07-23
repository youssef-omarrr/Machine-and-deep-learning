# Lecture 3: Deep Computer Vision
### [Video Link](https://www.youtube.com/watch?v=oGpzWAlP5p0&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=3&ab_channel=AlexanderAmini)
### [Slides Link](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L3.pdf)

# NOTES:
## Regression vs Classification

**Regression**: predicts **continuous numerical values** (e.g., house price, temperature). The output is a real number
**classification**: predicts **discrete class labels** (e.g., cat vs. dog).  The output is a class probability or category.

---
##  From Manual Features to Learned Features

- Traditional vision relied on crafting features manually (e.g., edges, textures).
    
- Drawbacks:
    
    - Requires domain expertise.
        
    - Doesn’t scale well.
        
- Deep learning **automatically learns a hierarchy**:
    
    - **Low-level**: edges, blobs.
        
    - **Mid-level**: motifs, textures.
        
    - **High-level**: object parts, full structures.
	    ![Alt text](imgs/Pasted%20image%2020250706215538.png)
        

---

## Fully Connected (Dense) Networks

- Every neuron is connected to all inputs → **no spatial awareness**.
    
- High parameter count → **inefficient** for images.

**Solution**: Use the **spatial structure of images** by applying **convolutional layers** instead of dense (fully connected) layers.

This reduces the number of parameters, **preserves local features**, and allows the model to **scale efficiently** to high-dimensional inputs like images.
![Alt text](imgs/Pasted%20image%2020250706215747.png)

---

##  Convolutional Neural Networks (CNNs)

### Key Properties:

1. **Local connectivity**: neurons only connect to a patch (receptive field).
    
2. **Shared weights (filters)**: same filter applied across the image.
    
3. **Multiple filters**: to detect varied features.
    

### Convolution operation:

- Slide filter over input, compute element-wise products and sum.
    
- Generates a **feature map**.
- Example of mapping features of the letter 'x' to be able to be detected even if rotated.![Alt text](imgs/Pasted%20image%2020250706215837.png)
    
- Example of the use of different filters to get different outputs.![Alt text](imgs/Pasted%20image%2020250706220012.png)
---

##  CNN Architecture Pipeline

1. **Convolution**: extract spatial features.
    
2. **ReLU** activation: introduce non-linearity. 
    $$ \text{ReLU}(x) = \max(0, x) $$
  
3. **Pooling** (e.g., MaxPool): reduce resolution while preserving key features.
    
4. Repeat layers (Conv → ReLU → Pool).
    
5. **Fully connected** layers & **Softmax** for classification.

![Screenshot 1](imgs/Pasted%20image%2020250706220151.png)
![Screenshot 2](imgs/Pasted%20image%2020250706220244.png)


---

##  Beyond Classification

- CNNs power:
    
    - **Semantic segmentation**, **object detection**, **image captioning**, **autonomous navigation**, **medical diagnostics**.
        
- Tools like **FCNs**, **R-CNN**, and **CNN + RNN architectures** are used.
    ![Alt text](imgs/Pasted%20image%2020250706220352.png)![Alt text](imgs/Pasted%20image%2020250706220415.png)

---
