## Sources
- [How to Train YOLO Object Detection Models in Google Colab (YOLO11, YOLOv8, YOLOv5)](https://www.youtube.com/watch?v=r0RspiLG260)

---

##  1. Install and Import

```bash
pip install ultralytics
```

Then import:

```python
from ultralytics import YOLO
import torch
print("GPU available:", torch.cuda.is_available())
```

---

##  2. Prepare Dataset

YOLO expects the dataset in this format:

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Each image in `images/train` must have a corresponding `.txt` file in `labels/train`
with YOLO annotations:

```
class_id x_center y_center width height
```

(all values normalized between 0–1).

### Example of `data.yaml`

```yaml
train: path/to/dataset/images/train
val: path/to/dataset/images/val

nc: 3
names: ['cat', 'dog', 'person']
```

---

##  3. Train YOLO on GPU

```python
from ultralytics import YOLO

# Load a pretrained model (transfer learning)
model = YOLO("yolov8s.pt")

# Train on your custom dataset
model.train(
    data="path/to/data.yaml",
    epochs=50,
    imgsz=640,
    device=0  # 0 = first GPU, "cpu" if no GPU
)
```

---

##  4. Evaluate the Model

```python
model.val()  # runs validation on val set
```

---

##  5. Run Inference on Test Images

```python
results = model("path/to/test_image.jpg", show=True)
```

---

##  6. Real-Time Detection with Webcam

```python
# Press 'q' in the window to quit
model.predict(source=0, show=True, conf=0.5)
```

---

##  7. Save Trained Weights

YOLO automatically saves everything (weights, metrics, plots) in:

```
runs/detect/train/
```

You can load your trained model later using:

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

---

##  Summary Pipeline

| Step            | Code                                   |
| --------------- | -------------------------------------- |
| Install         | `pip install ultralytics`              |
| Import          | `from ultralytics import YOLO`         |
| Prepare dataset | Create `data.yaml` + train/val folders |
| Train           | `model.train(...)`                     |
| Validate        | `model.val()`                          |
| Test            | `model("image.jpg", show=True)`        |
| Webcam          | `model.predict(source=0, show=True)`   |

---

