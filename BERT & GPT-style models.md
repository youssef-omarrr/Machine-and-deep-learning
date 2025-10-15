
### **BERT (Encoder-Only Model)**

* Based on the **Transformer encoder** stack.
* Uses **bidirectional self-attention**, every token attends to tokens **before and after** it.
* Trained with **Masked Language Modeling (MLM)**, randomly hides words and predicts them from context.
* Great for **understanding tasks**: classification, NER, question answering, etc.
* Output represents **contextual embeddings** for the input text.

**Key idea:** Deep *bidirectional* understanding of text.

---
### **GPT (Decoder-Only Model)**

* Based on the **Transformer decoder** stack.
* Uses **causal (unidirectional) self-attention**, each token attends only to **previous tokens**.
* Trained with **next-token prediction**, predicts the next word given all prior words.
* Great for **generation tasks**: text completion, dialogue, summarization, etc.
* Produces **autoregressive outputs**, one token at a time.

**Key idea:** Left-to-right generative language modeling.

---
### Summary

![](../imgs/PastedImage-41.png)

| Feature             | **BERT (Encoder)**             | **GPT (Decoder)**         |
| :------------------ | :----------------------------- | :------------------------ |
| Attention direction | Bidirectional                  | Unidirectional (causal)   |
| Training objective  | Masked Language Modeling       | Next-token prediction     |
| Strength            | Understanding / Representation | Generation / Continuation |
| Output type         | Context embeddings             | Generated text            |

---

