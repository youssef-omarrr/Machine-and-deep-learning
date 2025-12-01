# Conv1D vs Conv2D vs Conv3D

## The core idea

A **convolution layer** slides filters (kernels) across an input tensor in one or more spatial/temporal dimensions to detect patterns.

| Type       | What it’s used for                  | What it slides across                     |
| ---------- | ----------------------------------- | ----------------------------------------- |
| **Conv1D** | Time-series, audio, sensor data     | **1 dimension** (time or sequence)        |
| **Conv2D** | Images, spectrograms                | **2 dimensions** (height × width)         |
| **Conv3D** | Videos, volumetric data (e.g., MRI) | **3 dimensions** (depth × height × width) |

---

## 1️⃣ Conv1D — “along time”

**Input shape:**
`[batch, channels, time]`
(Example: 6 IMU channels over 1024 time steps)

**What happens:**

* Each kernel slides **only along the time axis**.
* It looks for temporal patterns *within each channel* and *across channels* (via depth).

**Example**

```python
nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5)
```

If your input is `[B, 6, 1024]`, the output becomes `[B, 64, 1020]` (or smaller depending on stride/padding).

→ **Used for:**
IMU, ECG, EEG, audio waveforms, sequences.

---

## 2️⃣ Conv2D — “along height & width”

**Input shape:**
`[batch, channels, height, width]`
(Example: RGB image with 3 channels)

**What happens:**

* Each kernel slides **horizontally and vertically** across the image.
* It captures spatial patterns (edges, textures, shapes).

**Example**

```python
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3))
```

If your input is `[B, 3, 224, 224]`, output is `[B, 64, 222, 222]`.

→ **Used for:**
Images, spectrograms, 2D signals.

---

## 3️⃣ Conv3D — “height × width × depth (time)”

**Input shape:**
`[batch, channels, depth, height, width]`
(Example: video frames)

**What happens:**

* Kernel slides **in 3 directions**.
* It captures spatio-temporal patterns (movement across frames).

**Example**

```python
nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3,3,3))
```

Input `[B, 3, 16, 112, 112]` → output `[B, 64, 14, 110, 110]`.

→ **Used for:**
Videos, medical 3D scans (MRI/CT), volumetric data.

---

## Visual intuition

| Type       | Think of it as                    | Example use                         |
| ---------- | --------------------------------- | ----------------------------------- |
| **Conv1D** | Sliding filter over **a line**    | IMU time-series (x-axis = time)     |
| **Conv2D** | Sliding filter over **a surface** | Image pixels (x & y)                |
| **Conv3D** | Sliding filter through **a cube** | Video or 3D scan (x, y, time/depth) |

---

## Example with your IMU data

For you:

* Each sample = `[1024, 6]` → time × channels
* You reshape to `[batch, 6, 1024]` for `Conv1D`.

Then:

```python
nn.Conv1d(6, 64, kernel_size=5, stride=2)
```

→ learns motion/tremor patterns *over time*, combining all IMU axes.

---

## Key takeaway

| CNN Type   | Input shape       | Learns across | Typical domain    |
| ---------- | ----------------- | ------------- | ----------------- |
| **Conv1D** | `[B, C, T]`       | Time          | IMU, ECG, Audio   |
| **Conv2D** | `[B, C, H, W]`    | Space         | Images            |
| **Conv3D** | `[B, C, D, H, W]` | Space + Time  | Videos / 3D scans |

---

### For your case

Use **Conv1D**, because:

* Your data are time-series (not images).
* You have 6 input channels (IMU axes).
* You want to extract temporal-frequency patterns.

---