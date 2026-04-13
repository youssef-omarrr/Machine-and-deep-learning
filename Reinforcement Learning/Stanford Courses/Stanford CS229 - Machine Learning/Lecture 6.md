# Naive Bayes (NB) — Summary

### Key assumptions

* **Conditional independence**: each feature ($x_j$) is independent given the class ($y$).
* **Ignores frequency**: repetition of features (like words) is not considered.
* **Ignores order**: the sequence of features doesn’t matter.

---

### Parameters

For binary features ($x_j \in {0,1}$) and binary class ($y \in {0,1}$):

| Parameter                                     | Meaning                                 |
| --------------------------------------------- | --------------------------------------- |
| $(\phi_{j\|y=1} = P(x_j = 1 \mid y = 1))$     | Probability feature (j) is 1 in class 1 |
| $(1 - \phi_{j\|y=1} = P(x_j = 0 \mid y = 1))$ | Probability feature (j) is 0 in class 1 |
| $(\phi_{j\|y=0} = P(x_j = 1 \mid y = 0))$     | Probability feature (j) is 1 in class 0 |
| $(1 - \phi_{j\|y=0} = P(x_j = 0 \mid y = 0))$ | Probability feature (j) is 0 in class 0 |

* These parameters are **learned using Maximum Likelihood Estimation (MLE)** from the training data.

---

### Problem: zero probability

* If a feature appears in **test data but not in training data**, its probability ($P(x_j) = 0$).
* This causes the model to assign **zero probability to the whole sample**, which is wrong (dividing by zero).

### Solution: **Laplace smoothing**

* Add 1 to the numerator (count of feature occurrences)
* Add 2 to the denominator (total count for binary feature)

Formula example:

$$
\hat{\phi}_{j|y=1} = \frac{\text{count}(x_j=1, y=1) + 1}{\text{count}(y=1) + 2}
$$

* Ensures **no probability is ever zero**, even for *unseen* features.

---
### Very simple word example

Suppose we are classifying emails as **spam (y=1)** or **not spam (y=0)**.

- Feature $x_{\text{“free”}}$​ = 1 if the word “free” appears in the email, 0 otherwise.
- Suppose in training data:

|Word|Spam (y=1)|Not spam (y=0)|
|---|---|---|
|free|3 / 4 emails|1 / 6 emails|

Then:

$$\phi_{\text{“free”}|y=1} = \frac{3+1}{4+2} = \frac{4}{6} = 0.667$$$$ \phi_{\text{“free”}|y=0} = \frac{1+1}{6+2} = \frac{2}{8} = 0.25$$

- If a new email contains the word “free”, it **increases the probability that it’s spam**, according to Naive Bayes.