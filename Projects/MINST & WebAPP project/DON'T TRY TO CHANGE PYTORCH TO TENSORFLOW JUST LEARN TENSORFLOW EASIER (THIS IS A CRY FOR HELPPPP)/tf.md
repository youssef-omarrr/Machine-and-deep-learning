#### 1. `tf_rep`

Created from:

```python
from onnx_tf.backend import prepare
tf_rep = prepare(onnx_model)
```

#### 2. `tf_model`

This is typically:

```python
tf_model = tf_rep.tf_module  # This extracts the actual TensorFlow model
```

---

### ✅ Difference between `tf_rep` and `tf_model`

| Concept   | `tf_rep`                                                | `tf_model`                                       |
| --------- | ------------------------------------------------------- | ------------------------------------------------ |
| Type      | `TensorflowRep` object (ONNX-TF wrapper)                | A `tf.Module` (or similar low-level TF model)    |
| Purpose   | Holds extra info to bridge ONNX to TF                   | The actual TensorFlow model for inference        |
| Usage     | `.run()` for predictions using ONNX input/output format | Use like any normal TF model: `tf_model(input)`  |
| Can save? | No (it's a wrapper)                                     | ✅ Yes — use `tf.saved_model.save(tf_model, ...)` |

---

### 🧠 So, when saving:

Save **`tf_model`** using:

```python
tf.saved_model.save(tf_model, "MNIST_tf")
```

> ✅ This writes a TensorFlow `SavedModel` folder (with `assets/`, `variables/`, `.pb`)

---

### 📦 When loading:

```python
loaded = tf.saved_model.load("MNIST_tf")
```

This returns a `_UserObject` or `tf.Module`:

* It **can be used for inference**
* ❌ But **can't be trained** using `.fit()` because it’s not a `tf.keras.Model`

---

### 🧩 Summary — what to do depending on your goal:

| Goal                         | What to Use                               | How to Save                             | How to Load                          |
| ---------------------------- | ----------------------------------------- | --------------------------------------- | ------------------------------------ |
| ✅ **Inference only**         | `tf_model = tf_rep.tf_module`             | `tf.saved_model.save(tf_model, "path")` | `tf.saved_model.load("path")`        |
| ✅ **Training / Fine-tuning** | Convert ONNX to `tf.keras.Model` manually | `model.save("path")`                    | `tf.keras.models.load_model("path")` |

---

### 🚨 TL;DR:

* `tf_rep` is a wrapper. Don’t save it.
* `tf_model` is the actual model. Save it using `tf.saved_model.save`.
* You **cannot continue training** unless you recreate the model as a `tf.keras.Model`.

---
