# Machine and Deep Learning Study Roadmap
> Due to `Git LFS` problems, all 67 commits that included the repo's history had to be deleted :(

## Overview
This roadmap includes resources, courses and their projects, and notes that I am following to build a strong foundation in machine and deep learning.

## Contents
- **Prerequisites:** Quick revision on `NumPy`, `Pandas`, and `Matplotlib`.  
- **MIT 6.S191:** Introduction to Deep Learning.  
- **PyTorch Course**  
- **Projects:** Contains most course projects; larger projects have their own repos linked below.  
- **LLM from Scratch Course**  
- **ITI Summer Training:** Machine Learning (supervised and unsupervised).  
- **Graduation Project Notes:** Research notes from my Bachelor’s degree project on Parkinson’s disease (not the actual project, but concepts explored and ML-related notes for future reference).  

## Resources & Links

### **Roadmap Video:**  
- [How to Learn Machine Learning in 2024 (7-step roadmap)](https://www.youtube.com/watch?v=jwTaBztqTZ0&list=PLZzRSwjUKZxnidL9CayMD8_UqTaPTWtcB&index=3) *(contains links to many resources)*  

### **Courses:**  
- [MIT 6.S191: Introduction to Deep Learning (highly recommended)](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)  
- [Full PyTorch Course](https://www.youtube.com/watch?v=V_xro1bcAuA&list=PLZzRSwjUKZxnidL9CayMD8_UqTaPTWtcB)  
  - [GitHub Repo](https://github.com/mrdbourke/pytorch-deep-learning?tab=readme-ov-file#course-materialsoutline)  
  - [Book Page](https://www.learnpytorch.io/00_pytorch_fundamentals/)  
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)  
- [Stanford Introduction to Machine Learning (free on Udacity)](https://www.udacity.com/enrollment/ud120)  
- [Statistical Learning with Python](https://www.youtube.com/playlist?list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ)  

### **Online Tutorials & Blogs:**  
- [All Machine Learning Algorithms Explained in 17 Min](https://www.youtube.com/watch?v=E0Hmnixke2g&list=PLZzRSwjUKZxnidL9CayMD8_UqTaPTWtcB&index=2)  
- [15 Machine Learning Lessons I Wish I Knew Earlier](https://www.youtube.com/watch?v=espQDESe07w&ab_channel=InfiniteCodes)  
- [Cracking Machine Learning Interview:](https://github.com/shafaypro/CrackingMachineLearningInterview) Repository to prepare for ML interviews, covering common questions asked by top companies.  
- [Great ML/DL YT channel (StatQuest with Josh Starmer)](https://www.youtube.com/@statquest)

### **Useful Websites:**  
- [Browse State-of-the-Art](https://paperswithcode.com/sota)  
- [Hugging Face Models](https://huggingface.co/models)  
- [`timm` (PyTorch Image Models) Library](https://github.com/huggingface/pytorch-image-models)  

## Projects & Practice
> - **Older projects** in this repo are mostly drafts and unpolished experiments.  
> - **More recent projects** were developed in their own dedicated repositories (including experiments and failed attempts), and are no longer included in this repo.
> - For complete projects, please check the linked repositories.  

### **Main Projects**
- [**Project Mozart:**](https://github.com/youssef-omarrr/Project-Mozart) Custom Transformer-based music generation.  
- [**Brain Tumor Segmentation:**](https://github.com/youssef-omarrr/Brain-Tumor-Segmentation) Deep learning model for semantic segmentation of brain tumors from MRI scans.  
- [**German Tutor:**](https://github.com/youssef-omarrr/German-Tutor) AI-powered German language learning assistant.  
- [**CelluScan:**](https://github.com/youssef-omarrr/CelluScan) Automated blood cell classification with Vision Transformer.  
- [**Real-Time Number Detector Web App:**](https://github.com/youssef-omarrr/MNIST_Web_APP) Trained on MNIST + custom data.  

### **PyTorch Course Tasks**
- Fashion MNIST Model (with confusion matrix, Chapter 3)  
- Multi-class Evaluation Model (Non-linear, Chapter 2)  
- Linear Regression Model (Chapter 1)  
- **Final Project:**  
  - [**Food MINI Model:**](https://github.com/youssef-omarrr/Food_MINI_model) Experiment tracking and model deployment.  

---
### **Machine Learning & Deep Learning Mastery Checklist**
#### 1. Mathematical Foundations (from college)

- [x] Linear Algebra (vectors, matrices, eigenvalues, SVD)
- [x] Calculus (gradients, Jacobians, chain rule)
- [x] Probability & Statistics (distributions, expectation, Bayes)
- [ ] Optimization (gradient descent, Adam, RMSprop, etc.) (next term)

#### 2. Classical Machine Learning

- [x] Linear Regression
- [x] Logistic Regression
- [x] Decision Trees / Random Forests
- [x] Boosting and Bagging
- [x] Support Vector Machines (SVM)
- [x] K-Nearest Neighbors (KNN)
- [x] Naive Bayes (Multinomial, Gaussian)
- [x] Gradient Boosting / XGBoost
- [x] K-Means & other clustering methods (Hierarchical clustering, DBSCAN)
- [x] Gaussian Mixture Models (GMM)
- [x] Dimensionality Reduction (PCA)
- [x] Bias–Variance Tradeoff
- [x] Cross-validation
- [x] Regularization (Ridge (L2) and Lasso (L1)) 
- [x] Evaluation Metrics (Accuracy, Precision, Recall, F1, ROC, AUC)
- [x] Feature Engineering / Normalization

#### 3. Core Deep Learning

- [x] Perceptron, Feedforward Networks
- [x] Activation Functions (ReLU, Sigmoid, Tanh, GELU, etc.)
- [x] Backpropagation
- [x] Weight Initialization (Xavier, Kaiming)
- [x] Dropout / BatchNorm / LayerNorm
- [x] Loss Functions (MSE, Cross-Entropy, BCE, etc.)
- [x] Learning Rate Schedules
- [x] Early Stopping
- [x] Gradient Clipping
- [x] Data Augmentation
- [x] Mixed Precision Training

#### 4. Convolutional Neural Networks (CNNs)

- [x] Convolution / Pooling / Padding
- [x] LeNet
- [x] AlexNet
- [x] VGG
- [x] ResNet
- [x] EfficientNet
- [x] Transfer Learning
- [x] Object Detection (YOLO)
- [x] Segmentation (U-Net, Mask R-CNN)

#### 5. Recurrent and Sequential Models

- [x] RNNs
- [x] LSTMs
- [x] GRUs
- [x] Sequence-to-Sequence Models
- [x] Attention Mechanisms
- [x] Encoder–Decoder Architectures

#### 6. Transformers and Attention Models

- [x] Self-Attention & Multi-Head Attention
- [x] Transformer Architecture (Encoder, Decoder)
- [x] Vision Transformer (ViT)
- [x] BERT & GPT-style models (encoder vs decoder-only)
- [x] Positional Encoding
- [x] Fine-tuning large models
- [x] Top-k, Top-p (nucleus) sampling

#### 7. Generative Models

- [x] Autoencoders (AE)
- [x] Variational Autoencoders (VAE)
- [x] Generative Adversarial Networks (GANs)
- [ ] Diffusion Models (DDPM, Stable Diffusion, ControlNet)
- [ ] Flow-based Models (RealNVP, Glow)

#### 8. Self-Supervised / Representation Learning

- [ ] Contrastive Learning (SimCLR, MoCo)
- [ ] Masked Autoencoders (MAE, BEiT)
- [ ] DINO / CLIP (cross-modal self-supervision)

#### 9. Reinforcement Learning (RL)

- [ ] Markov Decision Processes (MDPs)
- [ ] Policy Evaluation / Improvement
- [ ] Value Iteration / Policy Iteration
- [ ] Temporal Difference Learning
- [ ] Monte Carlo Methods
- [ ] Deep Q-Network (DQN)
- [ ] Double / Dueling DQN
- [ ] Policy Gradient Methods
- [ ] Actor–Critic
- [ ] PPO (Proximal Policy Optimization)
- [ ] DDPG / TD3 / SAC (continuous control)
- [ ] Model-Based RL (Dreamer, MuZero)
- [ ] Multi-Agent RL (MADDPG, QMIX)
- [ ] Offline RL / Imitation Learning
- [ ] RLHF (Reinforcement Learning from Human Feedback)

#### 10. AI Agents & Autonomous Systems

- [ ] ReAct (Reason + Act) architecture
- [ ] AutoGPT / BabyAGI
- [ ] Retrieval-Augmented Generation (RAG)
- [ ] LangChain or LlamaIndex for tool-using agents
- [ ] Vector DB Memory (FAISS, Chroma, Milvus)
- [ ] Planning + Reflection loops
- [ ] Integrating RL / feedback into agents
- [ ] Multi-agent coordination and communication

#### 11. Deployment & MLOps

- [x] Hugging Face deployment
- [ ] TorchScript / ONNX model export
- [x] FastAPI or Flask inference servers
- [x] Dockerization for reproducible deployment
- [x] GPU inference optimization (mixed precision, batching)
- [x] Model quantization / pruning
- [ ] Distributed training (PyTorch DDP / DeepSpeed)
- [x] CI/CD for models (GitHub Actions + versioning)
- [x] Monitoring / logging in production (W&B, Mlflow, TensorBoard)

#### 12. Practical / Projects

- [x] Build and train models from scratch
- [x] Reproduce architectures (ViT, Transformer, etc.)
- [x] Train a symbolic music generation Transformer
- [x] Visualize training metrics (TensorBoard, W&B)
- [x] Use pretrained models (Hugging Face)
- [x] Handle datasets and dataloaders efficiently
- [ ] Implement an RL environment and agent
- [ ] Build a tool-using AI agent with memory


