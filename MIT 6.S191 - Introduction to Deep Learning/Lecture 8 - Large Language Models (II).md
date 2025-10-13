#  Lecture 8: Large Language Models (II) 

### [Video Link](https://www.youtube.com/watch?v=_HfdncCbMOE&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=8&ab_channel=AlexanderAmini)

## 1. Post-training
![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-10.png)
- Post-training refers to any training phase **after the base pretraining** of the language model. It adjusts the model to align more with **human preferences, instructions, or safety constraints**. Common post-training phases include **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning from Human Feedback (RLHF)**.
    
- This stage helps move from a raw language model (which completes text generically) to a more **useful assistant-like model**. For example, GPT-3.5 to ChatGPT involved post-training steps that trained the model to follow instructions and respond helpfully.
    
- In practice, post-training drastically improves behavior and safety by aligning the model to desired outputs using curated datasets and feedback mechanisms.
---
## 2. Fine-tuning
![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-11.png)
- Fine-tuning means training a **pretrained model** further on a **specific dataset** for a **specific task**, such as medical Q&A, legal document summarization, or chat-based interactions.
    
- It updates some or all model weights to specialize in new behaviors. It can be full fine-tuning (entire model) or **parameter-efficient tuning** (e.g., LoRA, adapters).
    
- Fine-tuning lets models retain core language ability while becoming better at domain-specific tasks. However, it requires careful **hyperparameter tuning, regularization, and evaluation** to avoid overfitting or catastrophic forgetting.
---

## 3. Datasets
### **Diversity: General-purpose vs Specific**

- **General-purpose** datasets (like OpenWebText, The Pile) aim to capture a *broad* understanding of language across topics. These are used in **pretraining**.
    
- **Task-specific** or **domain-specific** datasets are *focused* (e.g., biomedical, legal, or customer support). They are used in fine-tuning or post-training to give the model **expertise** in certain tasks.

![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-12.png)
### **Data Formats: Instruction Data vs Preference Data**

- **Instruction data** teaches the model how to respond to tasks by showing pairs of prompts and desired outputs (e.g., "Translate this → Response"). It supports Supervised Fine-Tuning (SFT).
    
- **Preference data** includes multiple model outputs **ranked by quality**, enabling Reinforcement Learning from Human Feedback (RLHF) where the model learns to prefer more helpful/humane responses.
 ![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-13.png)

### **SFT Example: Instruction Following**

- Supervised Fine-Tuning often uses datasets like **Alpaca or OpenAssistant**, where models are trained to respond to questions or instructions. These datasets include thousands of prompt/response examples.
    
- It helps align the model to follow user prompts directly and perform useful tasks like summarization, question answering, or translation.
    

###  **Preference Example: UltraFeedback**

- UltraFeedback is a high-quality preference dataset used for **fine-tuning reward models**. It contains ranked outputs across many tasks to train models to identify and generate more helpful or harmless completions.
    
- It supports training reward models that guide the final model via **reinforcement learning (e.g., PPO)**.
    

### **Chat Templates**

- Chat templates define how user and assistant messages are structured in training and inference. Consistent formatting is crucial, especially for chat-based models (e.g., `<|user|> Hi <|assistant|> Hello!`).
    
- Different models (OpenChat, LLaMA, Mistral) may expect different formats, so tools like **Axolotl or FastChat** use templates to handle them.

---

## 4. Training
### **TRL (Transformers Reinforcement Learning)**

- A library built on Hugging Face Transformers for doing **RLHF**, particularly using **PPO (Proximal Policy Optimization)**. It's used to train models with reward models instead of hardcoded loss functions.
    
- TRL is crucial in the final alignment stage of models like ChatGPT and Claude.
    

###  **Axolotl**

- Axolotl is a configuration-based tool that simplifies **fine-tuning and instruction-tuning** of LLMs, especially using **LoRA**. It supports multiple backends and chat templates, and runs well on consumer hardware.
    
- Ideal for small-scale research, custom model training, and reproducibility.
    

###  **Unsloth**

- A library built for **efficient training** of models like LLaMA with very low VRAM usage. It can fine-tune 7B models with **1 GPU (12–16 GB)** by using smart memory tricks and 4-bit quantization.
    
- Great for individuals or small labs doing cost-effective fine-tuning on local machines.
---
### Supervised Fine-Tuning (SFT) techniques**

![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-14.png)

#### 1. **Full Fine-Tuning** – _16-bit precision (FP16)_

- **How it works:** Updates **all the weights** of the pretrained model directly using 16-bit floating point precision (FP16). It requires loading and modifying the entire model in memory.
    
- **Pros:**
    
    - ✅ **Maximizes quality** – The model can fully adapt to the new task, making this the most powerful fine-tuning method.
        
- **Cons:**
    
    - ❌ **Very high VRAM usage** – Training even a 7B model requires **more than 40 GB of GPU memory**, limiting it to expensive hardware.
        

---

#### 2. **LoRA (Low-Rank Adaptation)** – _16-bit precision (FP16)_

- **How it works:** Keeps the original model **frozen**, and adds small, trainable low-rank matrices **A and B** into *each layer*. Only these are updated, reducing memory and compute cost. The output of LoRA is added ("⊕") to the original model’s output.
    
- **Pros:**
    
    - ✅ **Fastest training** – Because only a few parameters are trained, training is much faster and consumes less compute.
        
    - ✅ **Efficient** – Memory usage is lower than full fine-tuning and can work on mid-range GPUs.
        
- **Cons:**
    
    - ❌ **Still has high VRAM usage** – The full model still runs in FP16, so it’s better than full fine-tuning but not extremely lightweight.
        

---

#### 3. **QLoRA (Quantized LoRA)** – _4-bit precision (NF4)_

- **How it works:** Same as LoRA but with a key difference: the base model is **quantized to 4-bit precision** (using a format like NF4), drastically reducing memory usage. LoRA adapters are trained on top of this compact base.
    
- **Pros:**
    
    - ✅ **Low VRAM usage** – You can fine-tune large models (e.g., LLaMA-7B) on consumer GPUs like a 12GB RTX 3060.
        
    - ✅ **Scalable** – Ideal for running and training multiple models or merging them later.
        
- **Cons:**
    
    - ❌ **Slightly degrades performance** – The compression introduces a trade-off in precision and quality compared to FP16.
        

---

#### Summary Table:

|Technique|Precision|Trains Full Model?|VRAM Usage|Quality|Speed|
|---|---|---|---|---|---|
|**Full Fine-Tuning**|FP16|✅ Yes|🔴 Very High|🟢 Best|🟡 Slow|
|**LoRA**|FP16|❌ No (only adapters)|🟠 Moderate|🟢 Very Good|🟢 Fastest|
|**QLoRA**|4-bit (NF4)|❌ No (only adapters)|🟢 Low|🟡 Slightly Lower|🟢 Fast|

---
### **Proximal Policy Optimization (PPO)** and **Direct Preference Optimization (DPO)**.

![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-15.png)

#### **Proximal Policy Optimization (PPO)** – _used in RLHF_

- **How it works:**
    
    - A **reward model** is trained first using **human preferences** (e.g., comparing which of two responses is better).
        
    - The **trained model** generates outputs for some prompt, which are scored by the reward model.
        
    - A **frozen copy** of the base model is used to compute **KL divergence** (a penalty for straying too far from original behavior).
        
    - PPO uses the reward and the KL divergence to adjust the trained model’s weights.
        
- **Pros & Cons:**
    
    - ✅ **Maximizes quality** – PPO produces very aligned, helpful, and nuanced models (e.g., ChatGPT).
        
    - ❌ **Very expensive & complex** – Needs multiple models (policy, reward, frozen reference), plus many GPU hours.
        

---

####  **Direct Preference Optimization (DPO)** – _simplified alternative_

- **How it works:**
    
    - Instead of training a separate reward model, DPO uses **pairwise preference data** (which output is preferred) directly.
        
    - Both the **trained model** and a **frozen base model** score the preferred and rejected outputs.
        
    - A **contrastive loss** is used to nudge the trained model to assign higher scores to preferred responses compared to rejected ones.
        
- **Pros & Cons:**
    
    - ✅ **Fast and cheap** – DPO doesn’t require a reward model or RL loop, so it's much more efficient.
        
    - ❌ **Lower quality** – It may not reach the fine-grained alignment quality of PPO-trained models.
        

---

#### Summary

|Feature|PPO|DPO|
|---|---|---|
|Uses reward model|✅ Yes|❌ No|
|KL penalty (staying close)|✅ Yes|❌ No (implicitly via contrast)|
|Complexity|❌ High|✅ Low|
|Cost|❌ Expensive|✅ Cheap|
|Alignment quality|✅ High|❌ Moderate|
|Used in|GPT, Claude (final alignment)|Open-source fine-tuning (LLaMA, etc.)|

---
### **Common Training Parameters**

- **Learning Rate**: Usually 1e-5 to 2e-4 depending on model size and LoRA strategy.
    
- **Batch Size**: Smaller for larger models (8-16); effective batch size can be increased with gradient accumulation.
    
- **Max Length**: Input sequence length (often 512–2048 tokens), depending on context.
    
- **Epochs**: Usually 3–5 for LoRA or SFT; more for small datasets, fewer for large ones.
    
- **Optimizer**: Common ones include AdamW, often with cosine or linear decay.
    
- **Attention**: Attention heads and context length are determined by architecture; training doesn't usually change this but impacts memory use.
---
## 5. Model merging (started as a joke lol)

- Model merging refers to **blending weights** from multiple fine-tuned models into a **single one**, sometimes without retraining. It allows combining strengths of different models (e.g., one good at English, another at French).
    
- Originally a meme, it now has practical use, such as merging a **coding model** with a **conversational model** to produce hybrids like “CodeLLaMA-Instruct.”
    
- Methods like **ModelSoup** and **DareTuning** try to combine checkpoints in parameter space while preserving performance.
- Also used to produce Bi-lingual models.
- Can be **cascaded** into many merges creating a merge tree.
---
## 6. Evaluation
### **General Benchmarks: MMLU, Open LLM Leaderboards**

- **MMLU** (Massive Multitask Language Understanding) tests reasoning across subjects like history, math, and law. It's used to assess general-purpose ability.
    
- Open LLM Leaderboards (by HuggingFace or LMSYS) use automated pipelines to benchmark models on common tasks like ARC, GSM8k, TruthfulQA, etc.
    

###  **Focused Benchmarks**

- These target specific capabilities: e.g., **math (GSM8k)**, **code (HumanEval)**, or **factuality (TruthfulQA)**.
    
- They help researchers identify a model's **strengths and weaknesses**, informing further fine-tuning or model selection.
    

###  **Judge LLMs**

- Instead of using human annotators to evaluate responses, LLMs can judge quality automatically. These are called "Judge LLMs" or "LLM Judges".
    
- Meta-evaluators (e.g., GPT-4 used to judge outputs from smaller models) are **cost-effective** and increasingly **accurate** with pairwise preference tasks.
---

## 7. Future trends

 ### **Scaling Test-Time Compute: Majority Vote**

- Run the same model multiple times with slightly different sampling (temperature, seed) and pick the **most frequent answer**. Reduces variance and increases reliability.
    
- Especially helpful in **low-confidence** situations (e.g., coding tasks or math).

![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-16.png)

---

### **Best-of-N Sampling**

- Generate **N completions**, then use a scoring function (e.g., log-probability or a reward model) to pick the **best** one.
    
- Improves quality without changing training, though it increases inference cost.

![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-17.png)

---

### **Beam Search**

- A deterministic search that **scores multiple candidate sequences at each step** and keeps the best `k` until completion.
    
- Produces higher-quality outputs but can be **less diverse** and more repetitive than sampling-based methods.

![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-18.png)

---

## Conclusion

- The modern LLM pipeline involves multiple steps: pretraining, post-training, fine-tuning, evaluation, and optimization.
    
- The ecosystem is evolving rapidly with tools like LoRA, Axolotl, and Unsloth making LLM development more accessible.
    
- Future progress is likely to come from **smarter inference techniques**, **better training data**, **modular model designs**, and **automated alignment tools** like Judge LLMs.

![](MIT%206.S191%20-%20Introduction%20to%20Deep%20Learning/imgs/PastedImage-19.png)