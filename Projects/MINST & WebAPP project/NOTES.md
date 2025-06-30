# Things learned in the project

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

Let me know if you want to visualize the normalized vs. unnormalized images!
