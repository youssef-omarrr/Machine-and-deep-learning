# Chapter 4 notes:

## [**SLIDES**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/04_pytorch_custom_datasets.pdf)
## [**Book link**](https://www.learnpytorch.io/04_pytorch_custom_datasets/)
## [**Collab link**](https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/04_pytorch_custom_datasets.ipynb)


### 🧠 What the label really is:

When using `torchvision.datasets.ImageFolder`, the **label** is just an integer representing the **class index** of the image.

If your folder structure looks like this:

```
dataset/
├── cats/
│   ├── cat1.jpg
│   └── cat2.jpg
├── dogs/
│   ├── dog1.jpg
│   └── dog2.jpg
```

Then `ImageFolder` **automatically assigns numeric labels**:

* `'cats'` → 0
* `'dogs'` → 1

So when you load a sample:

```python
img, label = train_data[0]
```

The `label` is:

* `0` if the image came from the `cats` folder
* `1` if it came from the `dogs` folder

And it stays the same **regardless of batch size**.

---

### 📦 In a `DataLoader`

When you wrap `ImageFolder` in a `DataLoader`, labels get batched:

```python
img, label = next(iter(train_dataloader))
```

* `img.shape` → `[batch_size, 3, H, W]`
* `label.shape` → `[batch_size]`
* `label[i]` is the class index for the `i`-th image in the batch.

---

### 🔢 Example with `batch_size = 4`

```python
label = tensor([0, 1, 1, 0])
```

This means:

* The first and fourth images in the batch are from class `0` (e.g., `'cats'`)
* The second and third are from class `1` (e.g., `'dogs'`)

---

### ✅ To see class names:

You can map the label to its **class name** with:

```python
class_names = train_data.classes  # ['cats', 'dogs']
print(class_names[label[0]])      # e.g., 'cats'
```

---

### TL;DR:

* ✅ The label is an **integer index** representing the class of the image.
* ✅ The number doesn’t change — it just gets grouped into a tensor with batch size.
* ✅ You can get the actual class name using `.classes`.

---

### 🔍 Why does `label.shape` change with the batch size?

When you create a `DataLoader` from a dataset like `ImageFolder`, and specify a `batch_size`, PyTorch **groups multiple samples together** into a batch. This affects the shape of both images and labels.

---

### ✅ Without batching (i.e., `batch_size=1`):

Each sample is returned as:

* `img.shape` → `[3, H, W]`
* `label` → `int`, so `label.shape` gives **an error** (integers have no `.shape`)

But PyTorch wraps it as a 1-element tensor when batching:

```python
img, label = next(iter(train_dataloader))
# When batch_size=1
print(img.shape)   # torch.Size([1, 3, H, W])
print(label.shape) # torch.Size([1])
```

---

### ✅ With `batch_size=N`:

Now you're getting `N` images and `N` labels in a batch:

* `img.shape` → `[N, 3, H, W]`
* `label.shape` → `[N]` (a 1D tensor of class indices, one per image)

So changing the `batch_size` changes the **first dimension** of both `img` and `label`.

---

### Example:

| `batch_size` | `img.shape`     | `label.shape` |
| ------------ | --------------- | ------------- |
| 1            | `[1, 3, H, W]`  | `[1]`         |
| 8            | `[8, 3, H, W]`  | `[8]`         |
| 32           | `[32, 3, H, W]` | `[32]`        |

---

### Summary:

* The **label shape** changes with `batch_size` because the `DataLoader` returns labels as a batched tensor (e.g., `[8]` for 8 labels).
* Each element in that label tensor corresponds to a label for one image in the batch.

---

## **Operator fusion** 
is a performance optimization technique used in compilers and deep learning frameworks where multiple operations (or "operators") are **combined into a single kernel or function**, reducing overhead and improving runtime efficiency.

---

### 🔧 In More Practical Terms:

Normally, in a neural network, operations are executed one after the other — like:

```text
x = relu(x)
x = batch_norm(x)
x = add(x, residual)
```

Without fusion, each of these operations would:

* Launch separately on the CPU/GPU
* Possibly read/write to memory multiple times

With **operator fusion**, these operations are **merged into one single computation** that:

* Executes in one kernel launch (on GPU)
* Keeps data in faster registers or shared memory
* Avoids intermediate memory operations

---

### 🧠 Why Use Operator Fusion?

* **Fewer memory reads/writes** → reduced memory bandwidth usage
* **Fewer kernel launches** → less launch overhead
* **Improved cache utilization**
* **Faster training and inference**, especially on hardware accelerators (like GPUs or TPUs)

---

### ✅ Example in Deep Learning

Suppose you're doing this:

```python
y = torch.relu(x)
z = y + 3
```

Instead of running two kernels (one for `ReLU`, one for `add`), operator fusion merges them into one kernel that applies both operations in a single pass.

Frameworks like:

* **TensorFlow XLA**
* **PyTorch JIT**
* **TVM**
* **ONNX Runtime**

…automatically fuse compatible operations during optimization passes.

---

### 🧪 Real-World Analogy

Imagine a sandwich shop:

* Without fusion: one person adds lettuce, another adds tomato, then another cuts the sandwich — 3 steps, 3 people.
* With fusion: one person does all 3 steps in one go — faster, fewer handoffs.

---

If you’re working with PyTorch and want to *see* fusion, you can try `torch.jit.trace` and `graph_for` to view the computational graph with fused ops.





