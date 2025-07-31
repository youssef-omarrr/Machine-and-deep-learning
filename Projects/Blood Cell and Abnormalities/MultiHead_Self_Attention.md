# Full Multihead Self Attention explaination

###  Step 1: Linear Projections

```python
Q = self.q_linear(X)
K = self.k_linear(X)
V = self.v_linear(X)
```

* Input `X` has shape **\[B, N, D]**

  * `B`: batch size
  * `N`: number of tokens (patches + class token)
  * `D`: embedding dimension (`hidden_dim`)

We apply a different linear transformation to make:

* Q (queries)
* K (keys)
* V (values)

Each still has shape: **\[B, N, D]**

---

###  Step 2: Split into Multiple Heads

```python
Q = Q.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
```

This reshapes `Q` to **\[B, num\_heads, N, head\_dim]**
We do this so each head gets a slice of the full embedding to process.

`view()` is used to reshape a tensor without copying its data. It’s like telling PyTorch:

> “Reinterpret the data in a different shape, but keep the same total number of elements.”

We repeat the same for `K` and `V`.

---

###  Step 3: Compute Attention Scores

```python
attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

Let’s break it down:

* `Q` shape: \[B, heads, N, head\_dim]
* `K.transpose(-2, -1)` shape: \[B, heads, head\_dim, N]
* `matmul`: computes attention scores between each token pair.

Result: **\[B, heads, N, N]**

We divide by √head\_dim to prevent very large values (helps softmax behave better).

---

###  Step 4: Apply Softmax

```python
attn_weights = self.softmax(attn_scores)
```

We apply softmax **across the last dimension**, so the weights sum to 1 across each token's attention to others.
Shape remains: **\[B, heads, N, N]**

---

###  Step 5: Compute Weighted Sum of Values

```python
attn_output = torch.matmul(attn_weights, V)
```

Each value vector is **weighted by attention**, giving the output for each token.
Shape: **\[B, heads, N, head\_dim]**

---

###  Step 6: Concatenate Heads Back

```python
attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, D)
```

We:

1. Move `heads` back to its original spot → \[B, N, heads, head\_dim]
2. Flatten `heads × head_dim` back to full embedding → \[B, N, D]

When you call `.contiguous()`, you're telling PyTorch to:

> Make a copy of the tensor's data in memory so that it's laid out in a contiguous (standard) order.

---

###  Step 7: Final Linear Layer

```python
output = self.out_linear(attn_output)
```

Apply one final linear layer to mix everything back together.

---

###  Summary of Shapes

| Step              | Shape                     |
| ----------------- | ------------------------- |
| Input X           | \[B, N, D]                |
| Q/K/V             | \[B, N, D]                |
| Q/K/V after split | \[B, heads, N, head\_dim] |
| Attention scores  | \[B, heads, N, N]         |
| After softmax     | \[B, heads, N, N]         |
| Weighted sum      | \[B, heads, N, head\_dim] |
| After concat      | \[B, N, D]                |
| Output            | \[B, N, D]                |

