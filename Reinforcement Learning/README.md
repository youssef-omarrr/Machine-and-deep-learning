# Reinforcement Learning & AI Agents Checklist

## Check list
#### 9. Reinforcement Learning (RL)

##### **Foundations**

- [ ] Markov Decision Processes (MDPs)
- [ ] Bellman Equations (Value / Q-function / Advantage function)
- [ ] Exploration vs. Exploitation (ε-greedy, softmax policies)
- [ ] Policy Evaluation & Policy Improvement
- [ ] Value Iteration / Policy Iteration

##### **Core RL Algorithms**

- [ ] Monte Carlo Methods (First-visit / Every-visit)
- [ ] Temporal Difference Learning (TD(0), TD(λ))
- [ ] SARSA / Q-Learning
- [ ] Deep Q-Network (DQN)
- [ ] Improvements: Double DQN, Dueling DQN, Prioritized Replay

##### **Policy Gradient & Actor–Critic Methods**

- [ ] Policy Gradient (REINFORCE)
- [ ] Actor–Critic (A2C, A3C)
- [ ] PPO (Proximal Policy Optimization)
- [ ] TRPO (Trust Region Policy Optimization)

##### **Continuous Control Algorithms**

- [ ] DDPG (Deep Deterministic Policy Gradient)
- [ ] TD3 (Twin-Delayed DDPG)
- [ ] SAC (Soft Actor–Critic)

##### **Model-Based & Advanced RL**

- [ ] Planning (Dyna-Q)
- [ ] Model-Based RL (Dreamer, MuZero-style)
- [ ] Hierarchical RL (Options Framework)
- [ ] Meta-RL (MAML, RL²)
- [ ] Multi-Agent RL (MADDPG, QMIX)
- [ ] Offline RL (CQL, Conservative Q-Learning)
- [ ] Imitation Learning (BC, GAIL)
- [x] RLHF (Reinforcement Learning from Human Feedback)

##### **Practical Skills**

- [x] Learn OpenAI Gym / Gymnasium environments (spaces, wrappers, vector envs)
- [ ] Implement RL algorithms from scratch in PyTorch:
  - [x] Q-learning (tabular)
  - [x] DQN (with replay + target network)
  - [x] REINFORCE
  - [x] Actor–Critic (A2C)
  - [ ] PPO (full implementation)
  - [ ] DDPG / TD3 / SAC
- [x] Learn stable-baselines3 usage and comparison
- [x] Logging RL experiments (TensorBoard, Weights & Biases)

---

#### 10. AI Agents & Autonomous Systems

##### **Agent Architectures**

- [ ] ReAct (Reason + Act)
- [ ] AutoGPT / BabyAGI-style loop
- [ ] Planning + Reflection loops
- [ ] Tool-using agent frameworks (LangChain / LlamaIndex)

##### **Memory & Retrieval**

- [ ] Retrieval-Augmented Generation (RAG)
- [ ] Vector Databases (FAISS, Chroma, Milvus)
- [ ] Long-term memory strategies (episodic, semantic)

##### **Integrating RL with Agents**

- [ ] RL-driven tool selection
- [ ] Feedback loops for improving agent behavior
- [ ] Multi-agent coordination & communication

---

## Video Lectures

### Practical Implementation

- [x] [**Reinforcement Learning in 3 Hours | Full Course using Python**](https://www.youtube.com/watch?v=Mut_u40Sqz4&list=PLZzRSwjUKZxm1smpyUzfgj7aLN8nupQhp&index=2). 
    - A crash course on implementation with **Stable Baselines** and **Gymnasium**. Covers **PPO, A2C, and DQN**.

### AI Agents & Protocols

- [ ] [**LangGraph Complete Course for Beginners – Complex AI Agents with Python**](https://www.youtube.com/watch?v=jGg_1h0qzaM&list=PLZzRSwjUKZxm1smpyUzfgj7aLN8nupQhp&index=4&t=3s).
    - Focuses on building stateful AI agents using the **LangGraph** framework.
- [ ] [**Guide to Agentic AI – Build a Python Coding Agent with Gemini**](https://www.youtube.com/watch?v=YtHdaXuOAks&list=PLZzRSwjUKZxm1smpyUzfgj7aLN8nupQhp&index=5).
    - A project-based guide for creating a functional, tool-using coding agent.
- [ ] [**Intro to MCP Servers – Model Context Protocol with Python Course**](https://www.youtube.com/watch?v=DosHnyq78xY&list=PLZzRSwjUKZxm1smpyUzfgj7aLN8nupQhp&index=6).
    - Introduces the **Model Context Protocol (MCP)** standard for structured agent communication.

### Foundational Theory

- [ ] [**Stanford CS234 | Reinforcement Learning | Spring 2024 | Emma Brunskill**](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX). 
    - A comprehensive, university-level course on the mathematical and algorithmic foundations of modern RL.
- [ ] [**DeepMind x UCL | Deep Learning Lecture Series 2021 | Intro to RL**](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm). 
    - Foundational lectures from DeepMind researchers on core concepts of Deep Reinforcement Learning.

***

## GitHub Repositories

### Core Reinforcement Learning Frameworks

- **[Learn Deep Reinforcement Learning in 60 days!](https://github.com/andri27-ts/Reinforcement-Learning?tab=readme-ov-file)** 
    - A structured Deep RL course with lectures and full code in **Python/PyTorch**. Covers **DQN, PPO, A2C, DDPG, Model-Based RL**.
- **[Implementation of Reinforcement Learning Algorithms](https://github.com/dennybritz/reinforcement-learning)**
    - Code implementations and exercises for **Sutton & Barto's** textbook and **David Silver's** lectures, using **Python/TensorFlow**.
- **[Modularized Implementation of Deep RL Algorithms in PyTorch](https://github.com/ShangtongZhang/DeepRL)** 
    - A clean, modular library for implementing various modern Deep RL algorithms using **PyTorch**.
- **[Minimal and Clean Reinforcement Learning Examples](https://github.com/rlcode/reinforcement-learning)** 
    - Highly readable implementations of classic RL algorithms, ideal for quickly understanding concepts.
- **[PyTorch implementation of reinforcement learning algorithms](https://github.com/Khrylx/PyTorch-RL)** 
    - Another quality repository providing clear implementations of RL algorithms using **PyTorch**.

### AI Agent Tools & Architectures

- **[Airweave: Context Retrieval for AI Agents across Apps & Databases](https://github.com/airweave-ai/airweave)** 
    - An open-source **Retrieval-Augmented Generation (RAG)** layer for agents via **REST API** or **MCP**.
- **[FastMCP v2](https://github.com/jlowin/fastmcp)** 
    - A fast, Pythonic library for easily building and interacting with **Model Context Protocol (MCP)** servers and clients.
- **[Agentic AI Crash Course](https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/free_courses/agentic_ai_crash_course)** 
    - A collection of resources and materials for learning the fundamentals of agentic AI design.
- **[Verifiers: Environments for LLM Reinforcement Learning](https://github.com/PrimeIntellect-ai/verifiers)** 
    - Specialized environments for applying RL to train LLMs, e.g., via **RLHF**.
