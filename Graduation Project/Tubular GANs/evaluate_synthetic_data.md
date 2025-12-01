# Part 1: `evaluate_synthetic.py` functions

### 1. **Row counts**

- **What it shows**: Size of real vs. synthetic datasets.
- **Best result**: Not really about quality, but synthetic usually should be larger than real (data expansion) without being trivially identical.

---

### 2. **Exact duplicate fraction**

- **What it shows**: How many synthetic rows are exact copies of rows in the real dataset.
- **Best result**: Close to **0.0**.

  - High values mean **memorization** (privacy risk, low generalization).

---

### 3. **Numeric Kolmogorov–Smirnov (KS) tests**

- **What it shows**: Compares the distribution of each numeric column (real vs. synthetic).
- **Best result**: KS statistic close to **0.0**.

  - Typically:

    - <0.1 → very good match
    - 0.1–0.3 → acceptable
    - \> 0.3 → mismatch

👉 **Explanation**: The **Kolmogorov–Smirnov test** is a statistical test that compares two probability distributions by measuring the largest difference between their cumulative distribution functions (CDFs). If the KS statistic is small, the two distributions are similar.

[![Kolmogorov–Smirnov Test Histogram](https://spss-tutorials.com/img/kolmogorov-smirnov-normality-test-what-is-it-histogram.png)](https://www.spss-tutorials.com/spss-kolmogorov-smirnov-test-for-normality/)

---

### 4. **Correlation matrix Mean Squared Error (MSE)**

- **What it shows**: How well the synthetic data preserves correlations between features.
- **Best result**: **0.0**.

  - <0.05 → excellent
  - 0.05–0.15 → moderate
  - \> 0.15 → poor preservation of relationships

---

### 5. **Classifier Two-Sample Test (Area Under the Curve, AUC)**

- **What it shows**: Trains a classifier to distinguish real vs synthetic. Reports AUC.
- **Best result**: Close to **0.5** (random guessing).

  - 0.5 → indistinguishable (perfect synthesis)
  - \> 0.7 → suspicious
  - 1.0 → synthetic is trivially fake

👉 **Explanation**: The **AUC** stands for **Area Under the Receiver Operating Characteristic (ROC) Curve**. It measures how well a classifier separates two classes (real vs. synthetic). AUC = 0.5 means the classifier is no better than flipping a coin, while AUC = 1.0 means perfect separation. For synthetic data, we want the model to *fail* at separation (so closer to 0.5 is better).

[![Perfect Classifier ROC Curve AUC](https://www.researchgate.net/profile/Luis-Montesinos/publication/352037456/figure/fig2/AS:1030667492147211@1622741687824/The-area-under-the-receiver-operating-characteristic-curve-AUC-A-perfect-classifier.png)](https://www.researchgate.net/profile/Luis-Montesinos/publication/352037456/figure/fig2/AS:1030667492147211@1622741687824/The-area-under-the-receiver-operating-characteristic-curve-AUC-A-perfect-classifier.png)


---

### 6. **Downstream Machine Learning (ML) task (Train on synthetic → Test on real)**

- **What it shows**: How well a model trained on synthetic generalizes to real data.
- **Best result**: Scores (accuracy, F1, etc.) close to training on real.

  - Ideally: synthetic-based performance ≥ 90% of real-based performance.

---

### 7. **Principal Component Analysis (PCA) overlap fraction**

- **What it shows**: In PCA space, what fraction of synthetic samples have a **nearest neighbor** that’s real.
- **Best result**: Higher is better, usually **>0.5**.

  - <0.3 means synthetic data lives in a very different region.

👉 **Explanation**: **Principal Component Analysis (PCA)** is a dimensionality reduction method. It projects data into a lower-dimensional space (keeping the most important variance). By comparing real vs. synthetic in PCA space, we can check whether the synthetic data “lives” in the same region of variation as the real data.

---

## ✅ Summary of “Best” Targets

- **Duplicates** → 0.0
- **Kolmogorov–Smirnov (KS) / Total Variation Distance (TVD)** → <0.1
- **Correlation Mean Squared Error (MSE)** → <0.05
- **Area Under Curve (AUC)** → \~0.5
- **Downstream ML task** → accuracy/F1 close to real-trained baseline
- **Principal Component Analysis (PCA) overlap** → >0.5

---

# Part 2: `sdmetrics` functions

### 🔹 `evaluate_quality(real_data, synthetic_data, metadata)`

- Quick one-liner function from **SDMetrics (Synthetic Data Metrics library)**.
- Runs the **default QualityReport** under the hood.
- Returns a **dict of scores (0–1)** showing:

  - **Column Shapes** → how well individual column distributions match.
  - **Column Pair Trends** → how well relationships between pairs of columns match.
  - **Overall Score** → average of the two.

---

### 🔹 `QualityReport`

- A more **detailed and customizable report object**.
- Lets you:
  - Call `.get_score()` → overall similarity (0–1).
  - Call `.get_properties()` → per-metric summary (Column Shapes, Column Pair Trends).
  - Call `.get_details(property_name)` → fine-grained view (per-column or per-column-pair scores).
- Useful for **debugging** which specific features or relationships are weak.

---

## In short:

- **`evaluate_quality`** = quick snapshot (overall + 2 categories).  
- **`QualityReport`** = deep dive into *why* the quality is high or low.  

---

# KS vs QualityReport

### 1. **KS Test (Kolmogorov–Smirnov Test)**

- A **statistical test** that compares two continuous distributions (real vs. synthetic).
- It measures the **maximum distance between their CDFs (Cumulative Distribution Functions)**.
- Output: a **D-statistic** and a **p-value**.
    - Small distance (high p-value) → distributions are similar.
    - Large distance (low p-value) → they differ.
    
- **Scope**: **1D only** (one column at a time).  
- Use when you want to check _“does this single feature look the same in synthetic vs. real?”_

---

### 2. **QualityReport (from SDMetrics)**

- A **comprehensive evaluation framework** for synthetic data.
- It includes _multiple statistical tests and metrics_, not just KS.
- It evaluates:
    - **Column Shapes** (distribution similarity for each column, internally it may use KS, Chi-Squared Test, TVD, etc. depending on type).
    - **Column Pair Trends** (relationships between pairs of columns, using correlation similarity, regression scores, etc.).
    
- Produces:
    - An **overall quality score** (0–1).
    - **Per-column details** and **per-pair details**.
    
- **Scope**: **1D + 2D** (distributions *and* relationships).

---

### ⚖️ TL;DR

- **KS** = Kolmogorov–Smirnov Test, only for numerical columns, only 1D distribution.  
- **QualityReport** = suite of evaluations (may include KS-like metrics internally), covering both columns and relationships.  

---

👉 If your goal is just to test _one numeric column_, use **KS**.  
👉 If your goal is to judge _overall synthetic dataset quality_, use **QualityReport** (or `evaluate_quality`).  

---
