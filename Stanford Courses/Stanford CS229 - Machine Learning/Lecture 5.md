# Discriminative vs Generative Learning — Summary Notes

# 1. Discriminative Learning Algorithms

### Goal

Model **directly** the *conditional probability*:
$$p(y \mid x;\theta)$$

### Key idea

We assume the *conditional distribution* belongs to an **exponential family**:

* **Model**:
  $$ y \mid x;\theta \sim \text{ExponentialFamily}(\eta) $$
* **Natural parameter**:
  $$ \eta = \theta^{T} x $$

### Explanation

* We don’t model how **x is generated**.
* We only focus on predicting **y given x**.
* We **maximize conditional likelihood**:
  $$\max_{\theta} \prod_{i} p(y^{(i)} \mid x^{(i)};\theta)$$

This leads to logistic regression, softmax regression, etc.

---

# 2. Generative Learning Algorithms

### Goal

Model the **joint distribution**:
$$p(x, y) = p(x \mid y), p(y)$$

Then compute:
$$p(y \mid x) = \frac{p(x \mid y) p(y)}{\sum_{y'} p(x \mid y') p(y')}$$

### Why?

* First model how **each class generates x**.
* To **model $x$ means:

> **You assume a *probability distribution* for how the input data $x$ is generated, usually separately for each class.**

In other words, you try to answer the question:

> _“If I know the class label $y$, what does a typical $x$ that class look like?”_
* Then use **Bayes’ rule** for classification.

---

# Gaussian Discriminant Analysis (GDA)

## When to use

* When **x is continuous** and roughly **Gaussian per class**.
* Special case of a generative model.

## Model assumptions

For a *binary classification* problem ($y \in {0,1}):$

1. Prior:
   $$ p(y=1) = \phi,\quad p(y=0)=1-\phi $$

2. Class-conditional densities:
   $$ x \mid y=0 \sim \mathcal{N}(\mu_0, \Sigma) $$
   $$ x \mid y=1 \sim \mathcal{N}(\mu_1, \Sigma) $$

### Explanation

* Each class has its own *mean* vector ($\mu$).
* **Both classes share the same covariance** ($\Sigma$).
* This gives a **linear decision boundary**.

---

# Parameter Fitting (Maximum Likelihood)

We estimate:

* ($\phi$) (class prior)
* ($\mu_0, \mu_1$) (means for each class)
* ($\Sigma$) (shared covariance)

## MLE solutions

1. Prior:
   $$ \phi = \frac{1}{m} \sum_{i=1}^m \mathbf{1}{y^{(i)} = 1} $$
   (fraction of examples labeled 1)

2. Means:
   $$ \mu_1 = \frac{\sum_{i: y^{(i)}=1} x^{(i)}}{\sum_{i} \mathbf{1}{y^{(i)}=1}} $$
   $$ \mu_0 = \frac{\sum_{i: y^{(i)}=0} x^{(i)}}{\sum_{i} \mathbf{1}{y^{(i)}=0}} $$
   (average of x for each class)
- The “indicator” $1{y(i)=k}$ is just a **counter**:  
	it adds 1 when the label equals $k$, and 0 otherwise.

3. Shared covariance:
   $$
   \Sigma = \frac{1}{m}
   \sum_{i=1}^m
   (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
   $$
   (average squared deviation from each class’s mean)

---

# Generative vs Discriminative — Key Difference

| **Discriminative Models**                                                                                        | **Generative Models**                                                                                                         |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Model only what we care about:** the probability of the label given the input.                                 | **Model everything:** how data is generated and how labels relate to data.                                                    |
| $$p(y \mid x)$$                                                                                                  | $$p(x, y) = p(x \mid y)p(y)$$                                                                                                 |
| **Use _conditional likelihood_:** choose parameters that make the correct labels most likely _given the inputs_. | **Use _maximum likelihood_ on the joint model:** choose parameters that make the whole dataset (inputs + labels) most likely. |
| We maximize:$$\prod_i p(y^{(i)} \mid x^{(i)})$$                                                                  | We maximize: $$\prod_i p(x^{(i)}, y^{(i)})$$                                                                                  |
| **Does not model how x is generated** → fewer assumptions, simpler model.                                        | **Must assume a distribution for ($x\mid y$)** (e.g., Gaussian), which is a stronger assumption.                              |
| **Usually performs better when you have lots of data** because the model focuses only on the decision boundary.  | **Works well in small-data settings** because the extra structure (modeling ($x\mid y$)) gives more information.              |
| **Examples:** logistic regression, softmax regression, SVMs.                                                     | **Examples:** Gaussian discriminant analysis (GDA), Naive Bayes.                                                              |
### In one sentence
- **Discriminative models:** learn the *boundary* between classes.
- **Generative models:** learn how each class *produces* data, then classify with Bayes’ rule.
- **Modeling $x$** = learning the *distribution* of the data itself.
- **Modeling $y|x$** = learning only how to *classify* the data.
- **Maximum likelihood (ML)** in generative models finds parameters that make the **entire data ((x,y))** most probable under the *joint* distribution (p(x,y)).  
- **Conditional maximum likelihood** in discriminative models finds parameters that make the **labels (y)** most probable **given the inputs (x)** under the *conditional* distribution (p(y|x)), without modeling how (x) is generated.
---
