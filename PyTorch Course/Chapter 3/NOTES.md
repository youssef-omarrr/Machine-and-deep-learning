# Chapter 3 notes:

## [**SLIDES**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/03_pytorch_computer_vision.pdf)
## [**Book link**](https://www.learnpytorch.io/03_pytorch_computer_vision/)
## [**Collab link**](https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb)
## [CNN Explainer website](https://poloclub.github.io/cnn-explainer/).

![TinyVGG architecture, as setup by CNN explainer website](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-cnn-explainer-model.png)

![example of going through the different parameters of a Conv2d layer](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-conv2d-layer.gif)
Here's a concise summary of the parameters of `nn.Conv2d`:

1. **Padding**: Adds values (commonly zeros) around the input to preserve spatial dimensions and improve performance. Zero-padding is widely used for its simplicity and efficiency.

2. **Kernel Size**: Defines the size of the filter sliding over the input. Smaller kernels capture finer details and support deeper networks, while larger kernels extract broader features but reduce dimensionality faster.

3. **Stride**: Determines the step size of the kernel movement. Smaller strides extract more features and result in larger outputs, while larger strides reduce output size and may lose detail.

---

## **Extra-curriculum:** 
> Lookup "most common convolutional neural networks", what architectures do you find? Are any of them contained within the [`torchvision.models`](https://pytorch.org/vision/stable/models.html) library? What do you think you could do with these?

### Summary

A wide range of convolutional neural network (CNN) architectures have been developed since the seminal LeNet-5 in 1998, each pushing the boundaries of depth, parameter efficiency, and representational power. Among the “most common” are LeNet-5, AlexNet, ZFNet, GoogLeNet/Inception, VGGNet, ResNet (and its variants ResNeXt, Wide ResNet), DenseNet, and more recent lightweight or efficient models such as SqueezeNet, MobileNet, and EfficientNet.

Many of these architectures—AlexNet, VGG, ResNet, Inception, DenseNet, SqueezeNet, MobileNet, EfficientNet, and several others—are directly implemented in PyTorch’s `torchvision.models` library, complete with optional pre-trained ImageNet weights ([PyTorch][1]). With these ready-to-use implementations, you can quickly:

* **Perform transfer learning**, fine-tuning on your own datasets for classification, detection, or segmentation.
* **Extract features** from intermediate layers for tasks like clustering, zero-shot retrieval, or as backbone encoders in more complex models.
* **Benchmark** new methods against standard baselines.
* **Deploy** efficient versions (e.g., MobileNet, EfficientNet) on edge devices for real-time inference.

---

### 1. Common CNN Architectures

Below is a non-exhaustive list of the most influential and widely used CNN designs:

* **LeNet-5** (1998): The pioneering CNN for handwritten digit recognition on MNIST, introducing convolution and pooling layers in a trainable, end-to-end pipeline ([TOPBOTS][2]).
* **AlexNet** (2012): Sparked the deep learning revolution by winning ILSVRC 2012, employing ReLU activations, dropout, and data augmentation on ImageNet ([Medium][3], [Medium][4]).
* **ZFNet** (2013): An AlexNet refinement that visualized intermediate filters and improved accuracy by tweaking filter sizes and strides ([Medium][4]).
* **GoogLeNet/Inception** (2014): Introduced “Inception modules” to capture multi-scale features efficiently, achieving state-of-the-art results with a 22-layer deep architecture ([Medium][4], [GeeksforGeeks][5]).
* **VGGNet** (2014): Demonstrated that stacking multiple small (3×3) convolutions yields deeper yet manageable networks (e.g., VGG-16, VGG-19) ([Medium][3], [Jeremy Jordan][6]).
* **ResNet** (2015): Made ultra-deep networks trainable via residual (skip) connections, with variants up to 152 layers ([GeeksforGeeks][5], [Jeremy Jordan][6]).
* **ResNeXt** (2017): Extended ResNet by introducing “cardinality” (parallel residual branches) for better accuracy/complexity trade-offs ([Jeremy Jordan][6]).
* **DenseNet** (2017): Connected each layer to every other layer in a feed-forward fashion, improving parameter efficiency and gradient flow ([Jeremy Jordan][6]).
* **SqueezeNet** (2016): Achieved AlexNet-level accuracy with \~50× fewer parameters via “fire modules” ([Wikipedia][7]).
* **MobileNet** (2017) & **MobileNetV2/V3**: Utilized depthwise separable convolutions and inverted residuals for lightweight mobile inference ([PyTorch][1]).
* **EfficientNet** (2019): Introduced a compound scaling method to uniformly scale network width, depth, and resolution, achieving state-of-the-art accuracy/efficiency trade-offs ([PyTorch][1]).

---

### 2. Architectures in `torchvision.models`

PyTorch’s `torchvision.models` library offers ready-made implementations (with optional pre-trained ImageNet weights) of most of the above architectures, including but not limited to:

* **Image Classification**:
  - AlexNet, VGG (11/13/16/19)
  - ResNet (18/34/50/101/152)
  - ResNeXt
  - Wide ResNet
  - DenseNet
  - SqueezeNet
  - Inception V3
  - GoogLeNet
  - MobileNet V2/V3
  - EfficientNet (B0–B7)
  - MNASNet
  - RegNet
  - ShuffleNet V2
  - VisionTransformer
  - ConvNeXt 
  - SwinTransformer 
  - MaxViT ([PyTorch][1])
  
* **Detection & Segmentation**:
  - Faster R-CNN
  - Mask R-CNN
  - RetinaNet
  - SSD
  - FCN
  - DeepLabV3
  - Semantic Segmentation variants in `torchvision.models.detection` and `torchvision.models.segmentation` ([PyTorch][1])

---

### 3. Potential Uses

1. **Transfer Learning & Fine-Tuning**
   Leverage ImageNet-pre-trained weights to accelerate convergence and boost performance on your specific classification or detection tasks.

2. **Feature Extraction & Representation Learning**
   Use intermediate feature maps as embeddings for image retrieval, clustering, or as backbones in unsupervised/self-supervised pipelines.

3. **Benchmarking & Research**
   Establish strong baselines for new architectures, regularization techniques, or optimization algorithms by comparing against standard CNNs.

4. **Edge & Mobile Deployment**
   Employ lightweight models (SqueezeNet, MobileNet, EfficientNet-Lite) for on-device inference with limited compute and memory.

5. **Multi-Modal & Hybrid Models**
   Integrate CNN backbones with transformers, graph networks, or recurrent networks for tasks spanning vision, language, and beyond.

By tapping into the breadth of architectures in `torchvision.models`, you can prototype and deploy state-of-the-art vision systems quickly, iterate on custom modifications, or use these backbones as building blocks for more sophisticated models—all while standing on the shoulders of decades of CNN research.

[1]: https://pytorch.org/vision/stable/models.html "Models and pre-trained weights — Torchvision 0.22 documentation"
[2]: https://www.topbots.com/important-cnn-architectures/?utm_source=chatgpt.com "4 CNN Networks Every Machine Learning Engineer Should Know"
[3]: https://medium.com/%40imjeremyhi/notable-cnn-architectures-and-how-they-work-dd46dea65671?utm_source=chatgpt.com "Notable CNN architectures. Including AlexNet, VGGNet, UNet"
[4]: https://medium.com/%40fraidoonomarzai99/common-cnn-architectures-in-depth-c312b975482f?utm_source=chatgpt.com "Common CNN Architectures In Depth | by Fraidoon Omarzai - Medium"
[5]: https://www.geeksforgeeks.org/convolutional-neural-network-cnn-architectures/?utm_source=chatgpt.com "Convolutional Neural Network (CNN) Architectures - GeeksforGeeks"
[6]: https://www.jeremyjordan.me/convnet-architectures/?utm_source=chatgpt.com "Common architectures in convolutional neural networks."
[7]: https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C?utm_source=chatgpt.com "卷积神经网络"

