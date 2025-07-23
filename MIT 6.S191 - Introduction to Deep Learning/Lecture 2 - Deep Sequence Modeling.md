# Lecture 2: Deep Sequence Modeling
### [Video Link](https://www.youtube.com/watch?v=GvezxUdLrEk&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2&ab_channel=AlexanderAmini)
### [Slides Link](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf)

# NOTES:
## INTRO:
- ![Alt text](imgs/Pasted%20image%2020250706182621.png)
- ![Alt text](imgs/Pasted%20image%2020250706182705.png)

---
## Recurrent Neural Networks (RNNs)
![Alt text](imgs/Pasted%20image%2020250706182750.png)
**RNN (Recurrent Neural Network)** is a type of neural network designed to handle **sequential data** (e.g., time series, text, audio).

> It uses loops in the network to allow **information to persist** across time steps.

---

###  RNN Architecture

Each RNN cell takes:

- An input at time step $t: x_t$
    
- A **hidden state** from the previous time step $h_{t-1}$
    
- Outputs a new hidden state $h_{t}$
    

### 🔁 Formula:
![Alt text](imgs/Pasted%20image%2020250706183649.png)

$$h_t = \tanh(W_{hh}*h_{t-1} + W_{xh}*x_t + b)$$

- $h_t$: hidden state at time $t$
    
- $x_t:$ input at time $t$
    
- $W_{hh}$, $W_{xh}$: weight matrices
    
- $b$: bias
    

---

### 📦 Output Variants

1. **Many-to-One**: e.g., Sentiment analysis
    
2. **Many-to-Many**: e.g., Machine translation
    
3. **One-to-Many**: e.g., Image captioning (via encoder-decoder)
    

---

### 🔄 Unrolling the RNN

RNNs are “unrolled” through time:

```markdown
x₁ → h₁ →  
x₂ → h₂ →  
x₃ → h₃ → ...  
```

Each hidden state $h_t$ is affected by all previous inputs:

```markdown
h_t = f(h_{t-1}, x_t)
```

---

### ⚠️ Challenges of RNNs

####  1. **Vanishing Gradients**

- Gradients shrink as they backpropagate through many time steps.
    
- Hard to learn long-range dependencies.
    

####  2. **Exploding Gradients**

- Gradients grow too large → unstable training.
    
- Solution: **gradient clipping**

#### 3. Encoding bottleneck
Due to fixed size vectors.

#### 4. Slow, no parallelization
#### 5. Not long memory


---

### ✅ Solutions to RNN Limitations

#### 🔹 LSTM (Long Short-Term Memory)

- Introduces **gates** to control information flow.
    
- Better at learning long-range dependencies.
    

#### 🔹 GRU (Gated Recurrent Unit)

- Simplified version of LSTM (fewer gates).
    
- Often performs similarly with fewer parameters.
    

---

### 🛠️ PyTorch Example

```python
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1)
output, hn = rnn(input_seq, h0)
```

- `input_seq`: shape [seq_len, batch, input_size]
    
- `h0`: initial hidden state [num_layers, batch, hidden_size]
    

---

###  Study Notes Summary

#### Concepts
- RNN processes sequences step-by-step
- Hidden state $h_t$ stores memory of past
- Vanishing gradients → can’t learn long-term dependencies
- LSTM & GRU solve this with gates

#### Applications
- Text generation
- Speech recognition
- Language translation
- Time series prediction

#### Practice Tips
- Train a character-level RNN
- Visualize hidden states over time
- Try RNN vs LSTM on sequence tasks
---
## Embedding
An **embedding** is a learned, dense vector representation of discrete data (like words, tokens, or items). It transforms indexes into **fixed size** vectors.

> Instead of one-hot encoding, embeddings map items into a continuous **low-dimensional space**.

---

### Why Use Embeddings?

- Reduce **dimensionality** (e.g., words → 300D instead of 10,000D)
    
- Capture **semantic meaning** and **relationships**
    
- Make inputs **trainable** and **learnable**
    

---
### PyTorch Example

```python
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)
output = embedding(torch.tensor([5]))  # returns the 300D vector for word index 5
```

---
## Backpropagation Through Time (BPTT)

![Alt text](imgs/Pasted%20image%2020250706184143.png)

**Backpropagation Through Time** is the adaptation of standard backpropagation used to train **Recurrent Neural Networks (RNNs)**.

It accounts for the **sequential nature** of RNNs by unrolling the network across time steps and computing gradients at each step.

---

### Unrolling the RNN

For a sequence of length T, the RNN is unrolled into a chain of T copies, one for each time step:

```markdown
x₁ → h₁ →  
x₂ → h₂ →  
x₃ → h₃ →  
... → h_T
```

Each hidden state depends on the previous one:

```markdown
h_t = f(h_{t-1}, x_t)
```

---

### How BPTT Works

1. **Forward pass**: Compute outputs for all time steps.
    
2. **Loss computation**: Calculate total loss across all time steps.
    
3. **Backward pass**: Backpropagate errors from time step T back to t = 1, updating weights shared across time.
    

Gradients are accumulated over time:
$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W}$$


---

### Truncated BPTT

Due to memory and computation limits, we often use **Truncated BPTT**, which backpropagates only over a limited number of steps kk instead of the full sequence TT.

This improves efficiency but may limit the ability to learn long-term dependencies.

---

### Challenges

- **Vanishing gradients**: Gradients shrink over long sequences → hard to learn long-term dependencies.![Alt text](imgs/Pasted%20image%2020250706184359.png)
    
- **Exploding gradients**: Gradients grow rapidly → causes instability. ![Alt text](imgs/Pasted%20image%2020250706184345.png)
    
- Solutions include:
    
    - Gradient clipping
        
    - LSTM/GRU cells
        
    - Layer normalization
        

---

### Summary
- BPTT is used to train RNNs by unrolling them in time
- Gradients are computed across all time steps
- Truncated BPTT is a practical alternative for long sequences
- Suffers from vanishing/exploding gradients
- Solved by gated architectures like LSTM and GRU

---
## Attention is all you need

The paper introduces the **Transformer** architecture — a model that **entirely replaces recurrence with attention mechanisms**, achieving state-of-the-art results in sequence modeling tasks like machine translation.

---
###  Core Ideas

#### 1. **Self-Attention Mechanism**

Allows each token in a sequence to **attend to all other tokens** — enabling the model to learn dependencies **regardless of distance**.

#### Self-attention formula:

$$\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$


Where:

- $Q$: Queries
    
- $K$: Keys
    
- $V$: Values
    
- $d_k$: dimension of the key vectors
    

---

#### 2. **Multi-Head Attention**

Runs multiple self-attention mechanisms in parallel and combines them — allowing the model to learn **different types of relationships**.

---

#### 3. **Positional Encoding**

Since the model has no recurrence or convolution, **positional encodings** are added to inputs to give a sense of order:


$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\ \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$


---

#### 4. **Encoder-Decoder Structure**

The Transformer consists of:

- **Encoder**: processes the input sequence
    
- **Decoder**: generates the output sequence, one token at a time, using attention over encoder outputs
    

Each layer is composed of:

- Multi-head attention
    
- Feed-forward neural network
    
- Residual connections + Layer normalization
    

---

###  Advantages Over RNNs

#### 1. **Parallelization**

- **RNNs process tokens sequentially**, so can't be parallelized across time.
    
- **Transformers process all tokens at once**, enabling fast training.
    

#### 2. **Long-Term Dependencies**

- RNNs struggle with **vanishing gradients** for long sequences.
    
- Transformers use self-attention to directly model **global context**, regardless of distance.
    

#### 3. **Efficiency**

- Transformers are more efficient on GPUs due to their **matrix-based operations**, while RNNs are slower due to their sequential nature.
    

#### 4. **Better Performance**

- Transformers outperform RNNs on tasks like:
    
    - Machine Translation (BLEU score)
        
    - Language modeling (BERT, GPT)
        

---

###  Summary Table


| Feature               | RNN                        | Transformer                  |
|-----------------------|----------------------------|------------------------------|
| Sequence Processing   | Sequential (one step at a time) | Fully parallel              |
| Long-term Dependencies| Hard to capture            | Handled via self-attention   |
| Training Time         | Slower                     | Much faster                  |
| Memory Requirements   | Lower                      | Higher, but manageable       |
| Positional Info       | Implicit (via time steps)  | Explicit (positional encoding) |


---

###  Final Takeaway

> The **Transformer** eliminates recurrence and uses **self-attention** to model sequence data. This makes it **faster**, more **parallelizable**, and better at capturing **long-range dependencies** than RNNs, leading to significant improvements in NLP tasks.

---
- Attention: How close my `Query` $Q$ is to the `Key` $k$ , then we extract the `Value` $v$.![Alt text](imgs/Pasted%20image%2020250706185249.png)
- This similarity is obtained by dot product![Alt text](imgs/Pasted%20image%2020250706185517.png) ![Alt text](imgs/Pasted%20image%2020250706185550.png) ![Alt text](imgs/Pasted%20image%2020250706185705.png) ![Alt text](imgs/Pasted%20image%2020250706185741.png)