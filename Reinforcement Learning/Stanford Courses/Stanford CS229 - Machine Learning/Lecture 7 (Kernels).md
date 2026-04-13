## Kernel Methods: The Core Idea

Kernel methods are a way to make **linear models** capable of solving **non-linear problems** without explicitly dealing with the often-massive dimension of the non-linear features.

The central idea is:
1.  **Map** the original data $\mathbf{x}$ into a high (possibly infinite) dimensional feature space using a **feature map** $\mathbf{\Phi}(\mathbf{x})$.
2.  Perform **linear learning** (like Linear Regression, SVM, etc.) in this new, high-dimensional space.
3.  Avoid the computational cost of the high-dimensional space by using the **Kernel Function** $\mathbf{K}(\mathbf{x}, \mathbf{z})$, which computes the **dot product** in that high-dimensional space implicitly.

### 1. The High-Dimensional Feature Space

Showing how a **non-linear** hypothesis can be **linear** in its parameters $\mathbf{\theta}$:

* **Hypothesis Function (Original space):**
    $$h_{\mathbf{\theta}}(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_1 x_2 + \dots$$
	- $x$: non-linear
	- $\theta$: linear
* **Feature Map ($\mathbf{\Phi}$):** This is the mapping that transforms the original input $\mathbf{x} \in \mathbb{R}^d$ into a new feature vector $\mathbf{\Phi}(\mathbf{x}) \in \mathbb{R}^p$.
    $$\mathbf{\Phi}(\mathbf{x}) = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ x_1^2 \\ x_1 x_2 \\ \vdots \end{bmatrix} \in \mathbb{R}^p$$
    Where $p$ can be **much, much larger** than $d$ ($p \gg d$). The original problem (with inputs $\mathbf{x}$) is now recast in the space of $\mathbf{\Phi}(\mathbf{x})$.

* **Hypothesis Function (Feature Space):** The model is **linear** in both $\mathbf{\theta}$ and $\mathbf{\Phi}(\mathbf{x})$.
    $$h_{\mathbf{\theta}}(\mathbf{x}) = \mathbf{\theta}^T \mathbf{\Phi}(\mathbf{x})$$

**The Problem:** If $p$ is huge (e.g., polynomial of degree 10 on 100 features, $p$ is astronomical), calculating $\mathbf{\Phi}(\mathbf{x})$ and training $\mathbf{\theta}$ (which has $p$ parameters) is computationally infeasible.

---

## 2. The Kernel Trick and Dual Formulation

### A. The Kernel Function

The **Kernel Trick** resolves the computational issue by introducing the Kernel function $K(\mathbf{x}, \mathbf{z})$.

* **Definition:** The Kernel function computes the *dot product* of two feature-mapped vectors **without** explicitly computing $\mathbf{\Phi}(\mathbf{x})$ or $\mathbf{\Phi}(\mathbf{z})$.
    $$K(\mathbf{x}, \mathbf{z}) = \langle \mathbf{\Phi}(\mathbf{x}), \mathbf{\Phi}(\mathbf{z}) \rangle = \mathbf{\Phi}(\mathbf{x})^T \mathbf{\Phi}(\mathbf{z})$$

* **The Key Insight (Representer Theorem):** For a wide range of *loss functions* and *regularization methods* used in ML (like Linear Regression with L2 regularization, or SVM), the **optimal solution** for the parameters $\mathbf{\theta}$ can **always** be expressed as a **linear combination** of the *feature vectors* of the training data:
    $$\mathbf{\theta} = \sum_{i=1}^{n} \beta_i \mathbf{\Phi}(\mathbf{x}^{(i)})$$
    Where $n$ is the number of training examples, and $\mathbf{\beta} \in \mathbb{R}^n$ are the new parameters (the **dual coefficients**).

### B. Predicting with the Kernel

By substituting the expression for $\mathbf{\theta}$ into the hypothesis $h_{\mathbf{\theta}}(\mathbf{x})$, the prediction depends entirely on $\mathbf{\beta}$ and the Kernel function.

* **The Prediction Equation:**
    $$h_{\mathbf{\theta}}(\mathbf{x}) = \mathbf{\theta}^T \mathbf{\Phi}(\mathbf{x}) = \left(\sum_{i=1}^{n} \beta_i \mathbf{\Phi}(\mathbf{x}^{(i)})\right)^T \mathbf{\Phi}(\mathbf{x})$$
    $$h_{\mathbf{\theta}}(\mathbf{x}) = \sum_{i=1}^{n} \beta_i \left(\mathbf{\Phi}(\mathbf{x}^{(i)})^T \mathbf{\Phi}(\mathbf{x})\right)$$
    $$h_{\mathbf{\theta}}(\mathbf{x}) = \sum_{i=1}^{n} \beta_i K(\mathbf{x}^{(i)}, \mathbf{x})$$

**Crucial Change:**
* Instead of optimizing $\mathbf{\theta} \in \mathbb{R}^p$ (where $p$ is huge), you optimize the $\mathbf{\beta} \in \mathbb{R}^n$ (where $n$ is the number of training examples).
* The computation during prediction/training only involves **dot products in the original space $\mathbb{R}^d$** or the result of the Kernel function $K(\cdot, \cdot)$, making it fast.

---

## 3. Important Kernels

### A. Polynomial Kernel

This is the explicit feature map you alluded to: $\mathbf{\Phi}(\mathbf{x})$ contains all polynomial terms up to degree $k$. The resulting kernel function is:

$$K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^k$$

For $c=0$, $k=2$, this represents the dot product in the space of all second-degree polynomial terms (squares and pairwise products) of $\mathbf{x}$ and $\mathbf{z}$.

### B. Gaussian / Radial Basis Function (RBF) Kernel

This is one of the most popular and powerful kernels. The $\mathbf{\Phi}(\mathbf{x})$ corresponding to this kernel is **infinite-dimensional**, yet the kernel computation is **very simple**. It measures the **similarity** between two inputs.

$$K(\mathbf{x}, \mathbf{z}) = \exp \left( - \frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2} \right)$$


* **Interpretation:** $K(\mathbf{x}, \mathbf{z})$ is high (close to 1) when $\mathbf{x}$ and $\mathbf{z}$ are close in the original space, and low (close to 0) when they are far apart.
* The parameter $\sigma^2$ (variance) controls the **spread** of the similarity. A small $\sigma^2$ leads to a narrow, highly localized similarity function.

### C. The Kernel Matrix $\mathbf{K}$ and Mercer's Theorem

To use the kernel trick, you first compute the **Gram matrix** (or **Kernel Matrix**) $\mathbf{K}$.

* **Definition:** The Kernel Matrix $\mathbf{K} \in \mathbb{R}^{n \times n}$ has elements given by:
    $$\mathbf{K}_{i,j} = K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$$
    This is *pre-computed* from the training data, taking $O(n^2 \cdot d)$ time.

* **Validity (Mercer's Theorem):** For a function $K(\mathbf{x}, \mathbf{z})$ to be a valid kernel (i.e., correspond to some dot product $\langle \mathbf{\Phi}(\mathbf{x}), \mathbf{\Phi}(\mathbf{z}) \rangle$), it is necessary and sufficient that the resulting Kernel Matrix $\mathbf{K}$ for *any* finite dataset $\{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)}\}$ must be **Positive Semidefinite (PSD)**.
    $$\mathbf{v}^T \mathbf{K} \mathbf{v} \ge 0 \quad \text{for all } \mathbf{v} \in \mathbb{R}^n$$

This theorem is what allows you to **design a kernel first** and then worry about $\mathbf{\Phi}(\mathbf{x})$ later (or not at all!).

---

## Why Kernels Are Less Used Today in Deep Learning

Kernels are less common in modern ML (Deep Learning) because of their $O(n^2)$ dependency.

| Kernel Method | Deep Learning |
| :--- | :--- |
| **Data Size ($n$)** | Requires matrix $K$, $O(n^2)$ complexity. Not scalable to millions of data points. | **$O(n)$** or better complexity per epoch. Highly scalable to massive datasets. |
| **Feature Space ($\mathbf{\Phi}$)** | Fixed by the Kernel function (e.g., polynomial, RBF). | **Learned** by the neural network's layers. Highly flexible and data-driven. |
| **Model Size ($p$)** | Size of $\mathbf{\theta}$ (if used) is huge $O(p)$. Size of $\mathbf{\beta}$ is $O(n)$. | Model size $O(p)$ is a user choice (number of parameters), typically much smaller than $n^2$. |

Kernel methods were crucial for tasks like image classification and text analysis before the rise of massive deep learning models, particularly with **Support Vector Machines (SVMs)**.

---
## TL;DR on Kernel Methods

Kernel Methods allow *linear* models to solve complex, *non-linear* problems by working in a **high-dimensional feature** space **implicitly**. The key is to *replace* the computation of the **high-dimensional features with a simple**, computationally cheap function: the **Kernel function**.

---

### Most Important Takeaways

- **The Kernel Trick is the substitute for $\mathbf{\Phi}$**: You **don't need to explicitly compute** the high-dimensional feature map $\mathbf{\Phi}(\mathbf{x})$. Instead, you use the **Kernel function** $K(\mathbf{x}, \mathbf{z})$, which calculates the dot product in the feature space: $K(\mathbf{x}, \mathbf{z}) = \mathbf{\Phi}(\mathbf{x})^T \mathbf{\Phi}(\mathbf{z})$.
- **Dual Formulation** The learning algorithm switches from optimizing the high-dimensional parameters $\mathbf{\theta}$ (size $p$) to optimizing the dual coefficients $\mathbf{\beta}$ (size $n$, the number of training examples).
- Prediction depends on Kernels and Training Data: A new prediction $h(\mathbf{x})$ is a weighted sum of kernels between the new input $\mathbf{x}$ and all training examples $\mathbf{x}^{(i)}$:
    
    $$h(\mathbf{x}) = \sum_{i=1}^{n} \beta_i K(\mathbf{x}^{(i)}, \mathbf{x})$$
    
- **Key Constraint (Mercer's Theorem)**: A function $K$ is a valid kernel only if its corresponding Gram matrix $\mathbf{K}$ (where $\mathbf{K}_{i,j} = K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$) is **Positive Semidefinite (PSD)** for any training set.
- **Scalability Issue**: The primary drawback is the complexity of computing and storing the Kernel Matrix $\mathbf{K}$, which is $O(n^2)$, making Kernel Methods (like Kernel SVM and Kernel Ridge Regression) impractical for large datasets ($n \gg 100,000$).
- **Popular Kernels**: The most important kernels are the **Polynomial Kernel** $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^k$ and the highly effective **Gaussian/RBF Kernel** $K(\mathbf{x}, \mathbf{z}) = \exp \left( - \frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2} \right)$.
---

## Dimensions

Let's re-examine the standard setup for **Linear Regression** *before* the Kernel Trick, and then look at the dimensions:

### 1. Standard Linear Regression Setup

| Variable | Description | Standard Size (Dimension) |
| :--- | :--- | :--- |
| $\mathbf{x}$ | Original input features | $d \times 1$ |
| $\mathbf{\theta}$ | Parameter vector (weights) | $d \times 1$ |
| $h_{\mathbf{\theta}}(\mathbf{x})$ | Hypothesis (Prediction) | $1 \times 1$ |

In this simple case, the hypothesis is $h_{\mathbf{\theta}}(\mathbf{x}) = \mathbf{\theta}^T \mathbf{x}$.

***

### 2. The High-Dimensional Feature Space Setup

When we use the **Feature Map $\mathbf{\Phi}$** to handle non-linearity, the dimensions change:

* **Original Input $\mathbf{x}$:** Still $\mathbf{x} \in \mathbb{R}^d$.
* **Feature Map $\mathbf{\Phi}(\mathbf{x})$:** Maps $\mathbb{R}^d \to \mathbb{R}^p$.
    $$\mathbf{\Phi}(\mathbf{x}) \in \mathbb{R}^p$$
    Where $p$ is the potentially very large number of high-order features (e.g., $1, x_1, x_2, x_1^2, x_1 x_2, \dots$). **So, $\mathbf{\Phi}(\mathbf{x})$ is of size $p$.**

* **Parameter Vector $\mathbf{\theta}$:** Since the hypothesis $h_{\mathbf{\theta}}(\mathbf{x}) = \mathbf{\theta}^T \mathbf{\Phi}(\mathbf{x})$ **must remain a scalar**, $\mathbf{\theta}$ must have the same dimension as $\mathbf{\Phi}(\mathbf{x})$.
    $$\mathbf{\theta} \in \mathbb{R}^p$$
    **Therefore, $\mathbf{\theta}$ is also of size $p$.**

**Your initial notes are correct in context:**

> * $\mathbf{\Phi}(\mathbf{x}) \in \mathbb{R}^p$ (Feature map of size $p$)
> * $\mathbf{\theta} \in \mathbb{R}^p$ (Parameter vector of size $p$)

The problem with this setup is that $p$ is often extremely large, making the optimization of $\mathbf{\theta}$ computationally expensive.

### 3. The Kernel Trick (Dual Formulation)

The Kernel Trick solves this by switching to the **dual parameters $\mathbf{\beta}$**.

* We no longer optimize $\mathbf{\theta} \in \mathbb{R}^p$.
* We now optimize the coefficients $\mathbf{\beta} \in \mathbb{R}^n$, where $n$ is the number of training examples. **This is why the Kernel Trick is powerful: it changes the dependence from the feature space dimension ($p$) to the number of data points ($n$).**

So, to summarize the dimensions used in the standard Kernel Method approach:

| Variable | Dimension | Role |
| :--- | :--- | :--- |
| $\mathbf{x}$ | $d \times 1$ | Original Input |
| $\mathbf{\Phi}(\mathbf{x})$ | $p \times 1$ | High-dimensional Feature Vector ($p$ is huge) |
| $\mathbf{\beta}$ | $n \times 1$ | Dual Coefficients (Optimized parameters) |
| $\mathbf{K}$ | $n \times n$ | Kernel Matrix (Gram Matrix) |

