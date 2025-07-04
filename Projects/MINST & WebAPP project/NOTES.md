# Things learned in the project

# `FormData()` Function

```javascript
const formData = new FormData();
```

### 🔍 What is `FormData`?

`FormData` is a built-in JavaScript object that lets you easily **construct key-value pairs** to send data — especially **files** — via an HTTP request (typically `POST`).

It mimics the way data is encoded when submitted using a standard HTML `<form>` with `enctype="multipart/form-data"`, which is the correct format for uploading files.

---

### ✅ Why do we use it here?

In your case, you're sending an **image file** (captured from a `<canvas>`) to a Flask server, so you need to package it correctly for the server to parse it as a file upload.

This line:

```javascript
formData.append('file', blob, 'image1.png');
```

does three things:

1. **`'file'`** → This is the key name your Flask server looks for (`request.files['file']`).
2. **`blob`** → This is the binary image data from the canvas.
3. **`'image1.png'`** → This is the file name assigned to the image. The Flask server doesn't strictly need it, but it helps mimic a real file upload.

---

### 🔄 Summary

| Part                                 | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `new FormData()`                     | Create a form-like object to hold fields and files |
| `append('file', blob, 'image1.png')` | Add your image to the form under the key `'file'`  |

This setup makes the `fetch()` request work just like submitting a form that uploads a file — so the Flask backend can handle it using:

```python
file = request.files['file']
```
---

# How **Flask receives** the image file sent from JavaScript using `FormData`.

### 🔁 Recap: What’s Being Sent?

In JavaScript:

```javascript
formData.append('file', blob, 'image1.png');

fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    body: formData
})
```

This sends a POST request with a file field named `"file"`.

---

### 🧠 Flask Side: How to Receive the File

In your Flask route:

```python
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']  # ✅ This retrieves the uploaded image file

    img = Image.open(file.stream)  # Open image using PIL
    ...
```

---

### 🔍 Breakdown of `request.files['file']`

| Part            | Purpose                                                               |
| --------------- | --------------------------------------------------------------------- |
| `request`       | The incoming HTTP request                                             |
| `request.files` | A special Flask dictionary-like object that stores all uploaded files |
| `'file'`        | The key used in JavaScript's `formData.append('file', ...)`           |
| `file.stream`   | A file-like object that PIL (Python Imaging Library) can read         |

You could also use:

```python
file.save('path/to/save.png')
```

if you wanted to store the image directly to disk instead of processing it.

---

### 🧪 Example of Error Handling

```python
if 'file' not in request.files:
    return jsonify({'error': 'No image file provided'}), 400
```

This checks if the key `'file'` is actually in the request, which protects against missing uploads or bad form submissions.

---

### 🧩 PIL Integration

You're using:

```python
img = Image.open(file.stream)
```

That line reads the image into memory using PIL (Python Imaging Library), so you can process it as a regular `PIL.Image.Image` object before feeding it to your PyTorch model.

---

### ✅ Summary

| JavaScript                      | Flask                                       |
| ------------------------------- | ------------------------------------------- |
| `formData.append('file', blob)` | `request.files['file']`                     |
| `fetch(..., body=formData)`     | `@app.route(..., methods=['POST'])`         |
| `image1.png` (filename)         | Available as `file.filename`                |
| `blob` (file content)           | Available as `file.stream` or `file.read()` |

---

# MNIST normalization

This line:

```python
transforms.Normalize((0.1307,), (0.3081,))
```

is part of a PyTorch preprocessing pipeline and is used to normalize input images. Specifically:

* It subtracts the mean (0.1307)
* Then divides by the standard deviation (0.3081)

for each pixel in the image (since MNIST images are grayscale, there's only one channel, hence the single-element tuples).

---

🧠 Why this matters

This standardization ensures your input data has:

* Mean ≈ 0
* Standard deviation ≈ 1

which:

* Helps the model converge faster
* Matches the distribution it saw during training (if your model was trained using these same normalization values)

—

🔍 Where do 0.1307 and 0.3081 come from?

These are the empirical mean and standard deviation of the MNIST training set (after scaling pixels to \[0, 1] by dividing by 255).

So:

* 0.1307 is the average pixel intensity
* 0.3081 is how much the pixel intensities vary

—

✅ Example: How this works

If a pixel has value 0.5 (scaled from 128 / 255), then:

normalized = (0.5 - 0.1307) / 0.3081 ≈ 1.2

So the image is now expressed in terms of “how many standard deviations from the mean” each pixel is.

—

✅ Important: Use same normalization for both training and inference

If your model was trained using Normalize((0.1307,), (0.3081,)), you must use the same during prediction.

If you’re debugging and want to see the unnormalized input, just skip this temporarily:

```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # Comment this out to view raw input
    # transforms.Normalize((0.1307,), (0.3081,))
])
```

---
