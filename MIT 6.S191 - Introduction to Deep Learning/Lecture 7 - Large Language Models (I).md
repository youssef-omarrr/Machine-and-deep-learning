# Lecture 7: Large Language Models (I) 

### [Video Link](https://www.youtube.com/watch?v=ZNodOsz94cc&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=7)

## 1. **Autoregressive Decoding**

- Autoregressive decoding is a method where a language model generates **one token at a time**, each conditioned on the *previous* tokens. For example, to generate "I am fine", the model first generates "I", then predicts "am" given "I", then "fine" given "I am", and so on.
    
- This approach is used in models like GPT, where output is sampled or selected from a **probability distribution** over the vocabulary at each step. It enables fluent, coherent text but is sensitive to early mistakes since future predictions depend on previous ones.
    

---

## 2. **Basic Language Model**

- A basic language model is trained to predict the *next word* in a sequence given the *previous words*. It learns the structure and semantics of a language purely from text data.
    
- These models can be autoregressive (like GPT), masked (like BERT), or sequence-to-sequence (like T5). They form the foundation of more advanced applications such as chatbots, translation systems, and summarizers.
    

---

## 3. **Building a Chatbot**

- To build a chatbot, you typically start with a pretrained language model and fine-tune or prompt it to generate relevant responses. You also need input/output handling, context management, and optionally memory or retrieval to improve response quality.
    
- Chatbots can be rule-based, retrieval-based (fetching answers from a database), or generative (producing responses like GPT). A well-built chatbot balances context understanding, appropriate tone, and response relevance.
    

---

## 4. **Zero-shot vs One-shot vs Few-shot**

- **Zero-shot**: The model is given only **a task instruction**, with *no* examples (e.g., “Translate to French: Hello”). It relies on its general language understanding.
    
- **One-shot**: The model is given *one example* along with the task instruction (e.g., “Translate to French: Goodbye → Au revoir. Translate to French: Hello →”).
    
- **Few-shot**: The model is provided with *several* examples (usually 3–10) to better understand the task. This can significantly improve performance on complex or unfamiliar tasks.
    

---

## 5. **How to Make It Better**

###  **Standard Prompting**

- This involves writing clear, concise task instructions and examples directly in the input to guide the model's output.
    
- Effective prompting is crucial, especially in zero/few-shot scenarios, and can drastically change the model’s performance.
    

### **Chain of Thought Prompting**

- Chain of Thought (CoT) prompting encourages the model to **"think aloud"** by including intermediate reasoning steps in its response.
    
- For example, instead of just giving an answer to a math question, the model explains its steps, improving accuracy in reasoning-heavy tasks.
    

### **Change the Network**

- **Adapters**: Lightweight modules inserted into the model’s layers; **only these are trained**, making it memory-efficient for task adaptation.
    
- **LoRA (Low-Rank Adaptation)**: Trains a **small set** of parameters by injecting low-rank matrices into the model's weights — highly efficient and used in many recent fine-tuning setups.
    
- **BitFit**: Fine-tunes only the bias terms of the model, making it extremely lightweight and quick to train, although with limited capacity.
    
- **Prompt/Prefix Tuning**: Instead of changing the model, you prepend learnable tokens (soft prompts) to guide behavior; useful for multi-task adaptation with minimal compute.
    
- **Ladder Side-Tuning**: Adds external layers ("side networks") that interact with different levels of the transformer, allowing multi-scale control over model behavior.
    

---

## 6. **How to Move Between Valid Language Models**

- Moving between models like GPT, LLaMA, or Mistral often involves *adapting* the prompting format, tokenizer, and model-specific quirks.
    
- You also need to evaluate capabilities (e.g., support for long context, instruction-following, tool use) and adjust input formatting accordingly.
    
- For consistent results, standardization libraries (like LangChain, OpenLLM, or HuggingFace Transformers) help abstract the differences between models.
    

---

## 7. **AI Agents**

###  **ReAct (Reason + Act)**

- ReAct is a **prompting** strategy that enables agents to reason *step-by-step* and take actions (like calling tools or APIs) in a loop.
    
- This allows LLMs to interact with their environment (e.g., a calculator or search engine) and use external information to complete tasks effectively.
    

###  **CoT (Chain of Thought)**

- CoT is a reasoning framework that makes the model generate intermediate reasoning steps before the final answer.
    
- It mimics human step-by-step logic and improves performance on tasks that require multi-step inference, arithmetic, or commonsense reasoning.
    

---
