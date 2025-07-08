# Lecture 1: Intro to Deep Learning
### [Video Link](https://www.youtube.com/watch?v=alfdI7S6wCY&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1&ab_channel=AlexanderAmini)
### [Slides Link](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf)

# NOTES:

## AI vs ML vs Dl:

- Artificial Intelligence (AI):
	-  **Definition**: AI is the broadest field. It refers to machines or software that can perform tasks that typically require **human intelligence**, such as reasoning, learning, planning, problem-solving, or language understanding.
	- **Goal**: To create intelligent agents.	    
	- **Examples**:
	    - Voice assistants like Siri or Alexa
	    - Game bots (e.g. in chess or Go)
	    - Recommendation systems
	    
- Machine Learning (ML):
	- **Definition**: ML is a **subset of AI**. It refers to algorithms that allow computers to **learn from data** and improve their performance over time without being explicitly programmed.
	- **Goal**: To enable systems to learn from experience.
	- **Examples**:
	    - Spam filters in email
	    - Predicting housing prices    
	    - Fraud detection
	    
- Deep Learning (DL):
	- **Definition**: DL is a **subset of ML** that uses **neural networks with many layers** (hence “deep”). It is particularly powerful for complex tasks involving images, sound, and text.
	- **Goal**: To model complex patterns in large amounts of data using layered neural networks.
	- **Examples**:
	    - Image recognition (e.g., identifying cats in photos)
	    - Speech recognition (e.g., Google Voice)
	    - Self-driving cars (e.g., recognizing traffic signs and pedestrians)
	    
- Think of it like this:
	- **AI** is the idea.
	- **ML** is how we achieve AI.
	- **DL** is a specific way of doing ML with powerful neural networks.
	
	---
## The Perceptron: Forward Propagation (one neuron)
![Alt text](imgs/Pasted%20image%2020250706163254.png)

---
## Non-linear (Activation functions):

- ✅ 1. **ReLU (Rectified Linear Unit)**
	**PyTorch**: `torch.nn.ReLU()` or `F.relu(x)`
	
	- **Formula**:
	$$f(x) = \max(0, x)$$
		
	- **Use**:  
		The most widely used activation in hidden layers due to its simplicity and effectiveness. Speeds up convergence and helps avoid vanishing gradients.
		
	- **Pros**: Fast, sparse activation (many zeros), avoids saturation for x>0
		
	- **Cons**: Can "die" if many outputs become zero (dying ReLU problem)
	
---
	
- ✅ 2. **Leaky ReLU**

	**PyTorch**: `torch.nn.LeakyReLU(negative_slope=0.01)` or `F.leaky_relu(x)`
	
	- **Formula**:
  	$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha*x & \text {otherwise} \end{cases}$$
	- **Use**:  
		Solves the “dying ReLU” problem by allowing a small, non-zero gradient when x<0x < 0.
		
	- **Common Slope**: α=0.01
	
---

- ✅ 3. **ELU (Exponential Linear Unit)**

	**PyTorch**: `torch.nn.ELU(alpha=1.0)` or `F.elu(x)`
	
	- **Formula**:
	$$f(x) = \begin{cases}x & \text{if } x > 0 \\\alpha (e^x - 1) &\text{otherwise}\end{cases}$$

	- **Use**:  
		Helps bring activations closer to zero mean, improving learning speed and robustness.
		
	- **Pros**: Smooth curve, non-zero gradient for x<0
	

---

- ✅ 4. **Sigmoid**

	**PyTorch**: `torch.nn.Sigmoid()` or `torch.sigmoid(x)`
	
	- **Formula**:
	$$f(x) = \frac{1}{1 + e^{-x}}$$

	- **Use**:  
		Often used in binary classification output layers. Maps input to range (0, 1).
		
	- **Cons**: Saturates at extremes; gradients vanish for large ∣x∣
	

---

- ✅ 5. **Tanh (Hyperbolic Tangent)**

	**PyTorch**: `torch.nn.Tanh()` or `torch.tanh(x)`
	
	- **Formula**:
  	$$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
	- **Use**:  
		Like sigmoid, but output range is (-1, 1), which helps with zero-centered data.
		
	- **Cons**: Also suffers from vanishing gradients
	

---

- ✅ 6. **Softmax**

	**PyTorch**: `torch.nn.Softmax(dim=1)` or `F.softmax(x, dim=1)`
	
	- **Formula**:
	$$f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

	- **Use**:  
		Converts logits into probabilities; used in the output layer of multi-class classification.
		
	- **Note**: Always used with `CrossEntropyLoss` (which includes `LogSoftmax` inside)
	

---

- ✅ 7. **GELU (Gaussian Error Linear Unit)**

	**PyTorch**: `torch.nn.GELU()` or `F.gelu(x)`
	
	- **Formula (approx)**:
	$$f(x) = 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

	- **Use**:  
		Used in Transformer models (like BERT). Smooth and probabilistic interpretation.
	

---

- ✅ 8. **Swish** _(not built-in, but used in many models)_

	- **Custom Implementation**:
	
	```python
	def swish(x):
		return x * torch.sigmoid(x)
	```
	
	- **Formula**:
  	$$f(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}$$
	- **Use**:  
		Proposed by Google, used in EfficientNet. Performs better than ReLU in many deep models.
    

---

### 🔍 Summary Table

|Activation|Range|Use Case|Pros|Cons|
|---|---|---|---|---|
|ReLU|[0, ∞)|Hidden layers|Fast, simple|Dying neurons|
|LeakyReLU|(-∞, ∞)|Same as ReLU, safer|Solves dying ReLU|Slightly slower|
|ELU|(-α, ∞)|Normalizing activations|Zero-centered, smooth|Slightly costlier|
|Sigmoid|(0, 1)|Binary classification|Probabilistic output|Vanishing gradients|
|Tanh|(-1, 1)|Normalized input data|Zero-centered|Vanishing gradients|
|Softmax|(0, 1), sum=1|Multiclass classification output|Probability vector|Not for hidden layers|
|GELU|(-∞, ∞)|Transformer-based architectures|Smooth, better convergence|Costlier than ReLU|
|Swish|(-∞, ∞)|EfficientNet and modern CNNs|Better than ReLU (empirically)|Not built-in (manual)|

---
## Gradient descent

- Gradient from math: is the direction of increasing losses, so we go opposite to this direction (back propagation) ![Alt text](imgs/Pasted%20image%2020250706165330.png)![Alt text](imgs/Pasted%20image%2020250706165354.png)![Alt text](imgs/Pasted%20image%2020250706165709.png)

---
## Back Propagation
- Back propagation is just chain rule
![Alt text](imgs/Pasted%20image%2020250706165557.png)

---
## Optimizer functions


- ✅ 1. **SGD (Stochastic Gradient Descent)**

	**Import**: `torch.optim.SGD`
	
	- **Use**: Classical optimizer used in early deep learning models and low-resource environments.
	    
	- **How it works**: Updates parameters using the gradient and a fixed learning rate.
	    
	- **Can include**:
	    
	    - **Momentum**: helps accelerate in the right direction.
	        
	    - **Weight decay**: adds L2 regularization.
	        
	
	```python
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	```
	
	- **Best for**: Simpler models like linear regression, small CNNs.
    

---

- ✅ 2. **Adam (Adaptive Moment Estimation)**

	**Import**: `torch.optim.Adam`
	
	- **Use**: Most popular optimizer for deep learning. Automatically adjusts learning rates using adaptive estimates of first and second moments (mean and variance).
	    
	- **Good default choice** for most problems.
	    
	
	```python
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	```
	
	- **Best for**: NLP, CNNs, Transformers, and when you're unsure what to use.
    

---

- ✅ 3. **AdamW (Adam with Weight Decay Fix)**
	
	**Import**: `torch.optim.AdamW`
	
	- **Use**: Improved version of Adam that **correctly implements weight decay**, unlike Adam which mixes weight decay with gradient updates.
	    
	- **Recommended for**: Training **transformers (e.g., BERT, GPT)** and large models.
	    
	
	```python
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
	```

---

- ✅ 4. **RMSprop**

	**Import**: `torch.optim.RMSprop`
	
	- **Use**: Maintains a moving average of the squared gradients, normalizing gradient steps.
	    
	- Often used in **recurrent neural networks (RNNs)** and **reinforcement learning**.
	    
	
	```python
	optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
	```
	
	- **Best for**: RNNs, noisy gradients, and non-stationary objectives.
    

---

- ✅ 5. **Adagrad**

	**Import**: `torch.optim.Adagrad`
	
	- **Use**: Adapts the learning rate to each parameter. Useful when dealing with **sparse data** (like NLP embeddings).
	    
	- **Downside**: Learning rate decays quickly and may stop learning.
	    
	
	```python
	optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
	```

---

- ✅ 6. **Adadelta**

	**Import**: `torch.optim.Adadelta`
	
	- **Use**: Extension of Adagrad that solves the learning rate decay issue.
	    
	- Eliminates the need to manually set the learning rate.
	    
	
	```python
	optimizer = torch.optim.Adadelta(model.parameters())
	```

---

- ✅ 7. **NAdam (Nesterov-accelerated Adam)**
	
	**Import**: `torch.optim.NAdam`
	
	- **Use**: Combines Adam with Nesterov momentum, leading to slightly better convergence in some tasks.
	    
	
	```python
	optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
	```

---

- ✅ 8. **ASGD (Averaged SGD)**

	**Import**: `torch.optim.ASGD`
	
	- **Use**: Like SGD but uses averaged weights to stabilize learning.
	    
	- Rarely used, but can improve generalization in some cases.
	    
	
	```python
	optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)
	```

---

### 🔍 Summary Table

|Optimizer|Use Case|Notes / Best For|
|---|---|---|
|**SGD**|Simple models, controlled updates|Add momentum for faster convergence|
|**Adam**|Most tasks, especially deep networks|Default choice; adaptive learning rates|
|**AdamW**|Transformers, large models|Better weight decay regularization|
|**RMSprop**|RNNs, reinforcement learning|Smooths out noisy gradients|
|**Adagrad**|Sparse data (e.g. NLP)|Learns fast initially, slows later|
|**Adadelta**|Same as Adagrad but better long-term|No need to set learning rate|
|**NAdam**|Deep networks (experimental)|Slightly better than Adam in some cases|
|**ASGD**|When stability/generalization matters|Averages model weights over time|

---

### 🔍 What is Weight Decay?

Weight decay is a regularization technique that modifies the loss function:

$$L_{\text{total}} = L_{\text{original}} + \lambda \cdot \frac{1}{2} \|w\|^2$$

Where:

- $L_{\text{original}}$ is your normal loss (e.g. cross-entropy).
    
- $w$ are the model's weights.
    
- $λ$ is the weight decay coefficient.
    

This encourages the optimizer to prefer smaller weights.

---

### ❌ Problem with Regular `Adam`

In standard `torch.optim.Adam`, **weight decay is implemented by adding the L2 penalty directly to the gradients**:

```python
param.grad += weight_decay * param
```

This **interferes with Adam's adaptive moment estimates**, because Adam uses momentum-like terms (mean and variance of gradients), and mixing the regularization directly into the gradients breaks the assumptions.

---

### ✅ AdamW Fixes This

**AdamW (Weight-decoupled Adam)** separates weight decay from gradient updates:

```python
# AdamW applies weight decay like this:
param -= learning_rate * weight_decay * param  # Separate from gradients
```

This keeps the optimizer's adaptive behavior **intact**, and applies regularization **cleanly** and **correctly**.

---

### 🔧 Summary

|Optimizer|Weight Decay Behavior|
|---|---|
|`Adam`|Adds decay to gradient (not ideal)|
|`AdamW`|Applies decay directly to weights (correct)|

---

## Regularization Techniques

What is Regularization?
**Regularization** is a set of techniques used to **prevent overfitting** by discouraging overly complex models.

> It helps your model **generalize** better to unseen data.

---
### ✅ 1. **L1 Regularization (Lasso)**

- **Penalty Term**:
    $$\lambda \sum |w|$$
- **Effect**: Drives some weights to **exact zero**, leading to **sparse models**.
    
- **Use Case**: When feature selection or sparsity is desirable.
    

---

### ✅ 2. **L2 Regularization (Ridge) / Weight Decay**

- **Penalty Term**:
    $$\lambda \sum w^2$$
- **Effect**: Penalizes large weights, but keeps all weights non-zero.
    
- **Use Case**: Most common regularization (used in `weight_decay` in optimizers like `AdamW`).
    

---

### ✅ 3. **Dropout**
![Image](imgs/Pasted%20image%2020250706171301.png)

- **How it works**: Randomly "drops" (sets to zero) a fraction of neurons during training.
    

```python
nn.Dropout(p=0.5)
```

- **Effect**: Prevents neurons from co-adapting too much; acts like ensemble averaging.
    
- **Use Case**: Very effective in fully connected layers and CNNs.
    

---

### ✅ 4. **Early Stopping**
![Image](imgs/Pasted%20image%2020250706171328.png)


- **How it works**: Stop training when validation loss stops improving.
    
- **Effect**: Prevents overfitting by avoiding unnecessary additional training.
    
- **Use Case**: Helps find the "sweet spot" before the model starts memorizing.
    

---

### ✅ 5. **Data Augmentation**

- **How it works**: Apply random transformations to training data (e.g., rotation, crop, noise).
    
- **Effect**: Increases data diversity, reducing overfitting.
    
- **Use Case**: Especially useful in image, audio, and text datasets.
    

---

### ✅ 6. **Batch Normalization (indirect regularization)**

- **How it works**: Normalizes layer inputs to have zero mean and unit variance.
    
- **Effect**: Stabilizes training, sometimes reduces overfitting.
    
- **Use Case**: CNNs, Transformers, and almost all modern deep nets.
    

---

### ✅ 7. **Label Smoothing**

- **How it works**: Instead of hard 0/1 labels, use soft targets like 0.9/0.1.
    
- **Effect**: Makes the model less confident and more robust.
    
- **Use Case**: Classification tasks, especially in NLP or vision.
    

---

### 🔚 Summary Table


| Technique        | Purpose                  | Effect                            |
|------------------|---------------------------|------------------------------------|
| L1               | Sparsity                 | Zero out unnecessary weights       |
| L2 / Weight Decay| Smoothness               | Penalize large weights             |
| Dropout          | Redundancy reduction     | Randomly disables neurons          |
| Early Stopping   | Generalization           | Stops before overfitting kicks in  |
| Data Augmentation| Data diversity           | Makes model robust to variations   |
| BatchNorm        | Training stability       | Faster, smoother convergence       |
| Label Smoothing  | Reduce overconfidence    | Softer, more calibrated predictions|


---
