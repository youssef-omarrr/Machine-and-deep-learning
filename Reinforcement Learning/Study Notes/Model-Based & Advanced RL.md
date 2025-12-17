# Model-Based & Advanced Reinforcement Learning

---

## 1. Planning (Dyna-Q)

### Motivation

Classic RL methods (Q-learning, SARSA) learn **only from real experience**, which can be:

* Sample inefficient
* Slow in environments where data is expensive

**Planning** integrates **learning + simulated experience**.

---

### Core Idea

Dyna-Q combines:

1. **Model-free learning** (Q-learning)
2. **Model learning** (learn environment dynamics)
3. **Planning** (simulate experience using the learned model)

The agent:

* Interacts with the real environment
* Learns a model: $$\hat{P}(s'|s,a), \hat{R}(s,a)$$
* Uses the model to generate *imaginary transitions*
* Updates Q-values from both real and simulated data

---

### Algorithm Flow

For each real step:

1. Observe $(s, a, r, s')$
2. Update Q-value:
   $$
   Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma \max_a Q(s',a) - Q(s,a)\big]
   $$
3. Update the **learned model**:
   $$
   Model(s,a) \leftarrow (r, s')
   $$
4. **Planning loop** (repeat $n$ times):

   * Sample $(\tilde{s}, \tilde{a})$ from past experience
   * Query model → $(\tilde{r}, \tilde{s}')$
   * Perform Q-learning update using simulated transition

---

### Key Insights

* Planning accelerates learning
* Model errors matter but often acceptable early
* Bridges model-free and model-based RL

---

### Advantages

* Sample efficient
* Simple to implement
* Improves convergence speed

### Limitations

* Requires discrete or manageable state-action space
* Learned model may be inaccurate

---

### Summary

Dyna-Q integrates **learning, modeling, and planning** by using a learned environment model to *generate simulated experiences*, significantly improving sample efficiency while retaining the simplicity of Q-learning.

---

## 2. Model-Based RL (Dreamer, MuZero-style)

### Motivation

Model-free deep RL:

* Requires millions of samples
* Is inefficient in real-world tasks

Model-Based RL aims to **learn dynamics** and **plan inside the model**.

---

## General Model-Based RL Structure

1. Learn a **world model**
2. Use the model to:
   * Plan actions
   * Learn value functions
   * Optimize policies

---

## Dreamer

### Core Idea

Dreamer learns a **latent dynamics model** and performs RL **entirely in imagination**.

---

### Components

1. **Encoder**
   $$
   z_t = f_{\text{enc}}(o_t)
   $$
2. **Latent Dynamics Model**
   $$
   z_{t+1} \sim p(z_{t+1} | z_t, a_t)
   $$
3. **Reward Model**
   $$
   r_t = f_r(z_t, a_t)
   $$
4. **Policy & Value Network**
   Trained using imagined rollouts

---

### Key Insight

Instead of predicting pixels:

* Dreamer learns **compressed latent dynamics**
* Planning happens in latent space

---

### Advantages

* Extremely sample efficient
* Works with high-dimensional observations (images)
* Stable training

---

## MuZero

### Core Idea

MuZero learns:

* **Dynamics**
* **Reward**
* **Value**
  Without knowing environment rules

---

### Architecture

1. **Representation**
   $$
   h_0 = f(o_0)
   $$
2. **Dynamics**
   $$
   (h_{t+1}, r_t) = g(h_t, a_t)
   $$
3. **Prediction**
   $$
   (v_t, \pi_t) = f_p(h_t)
   $$

---

### Planning

* Uses **Monte Carlo Tree Search (MCTS)**
* Searches over latent states
* Optimizes value and policy jointly
#### **Monte Carlo Tree Search (MCTS)** 
- Is a *planning* algorithm used to select actions by building a *partial search tree* through randomized simulations, making it especially effective in large or complex decision spaces where exhaustive search is infeasible. 

- MCTS iteratively performs four phases: 
	- **Selection**: where a tree policy such as UCB balances *exploration* and *exploitation* to traverse the current tree.
	- **Expansion**: where *new child nodes are added* when an unvisited state-action pair is encountered.
	- **Simulation (rollout)**: where actions are sampled, often randomly or using a learned policy, until a terminal state or depth limit is reached.
	- **Backpropagation**: where the obtained reward is propagated up the tree to update visit counts and value estimates.

By repeating these simulations, MCTS converges toward optimal decisions without requiring a full environment model, and it can be enhanced with neural networks for policy and value guidance, as in AlphaZero and MuZero.

---

### Key Difference (Dreamer vs MuZero)

| Aspect   | Dreamer             | MuZero            |
| -------- | ------------------- | ----------------- |
| Planning | Policy optimization | MCTS              |
| Training | Latent imagination  | Tree search       |
| Use case | Continuous control  | Games (Go, Atari) |

---

### Summary

Model-based RL methods like **Dreamer and MuZero** learn internal world models and perform learning or planning within them, achieving dramatic improvements in sample efficiency over model-free methods.

---

## 3. Hierarchical RL (Options Framework)

### Motivation

Flat RL struggles with:

* Long horizons
* Sparse rewards

Humans solve tasks hierarchically.

---

### Options Framework

An **option** is a temporally extended action:
$$
o = (\mathcal{I}, \pi_o, \beta)
$$
Where:

* $\mathcal{I}$: initiation set
* $\pi_o$: intra-option policy
* $\beta(s)$: termination probability

---

### Two-Level Policy

1. **High-level policy** selects options
   $$
   \pi(o|s)
   $$
2. **Low-level policy** executes primitive actions
   $$
   \pi_o(a|s)
   $$

---

### Example

Task: Navigation

* Option 1: Go to door
* Option 2: Open door
* Option 3: Enter room

---

### Learning

* Learn option policies
* Learn termination
* Learn option selection policy

---

### Benefits

* Temporal abstraction
* Better exploration
* Reusability of skills

---

### Challenges

* Discovering good options
* Credit assignment across levels

---

### Summary

Hierarchical RL decomposes complex tasks into reusable **options**, enabling efficient learning over long time horizons and sparse rewards.

---

## 4. Meta-RL (MAML, RL²)

### Motivation

Standard RL:

* Learns one task
* Fails to generalize

Meta-RL learns **how to learn**.

---

## MAML (Model-Agnostic Meta-Learning)

### Core Idea

Learn parameters $\theta$ that adapt quickly with few gradient steps.

---

### Inner Loop (Task Adaptation)

$$
\theta'*i = \theta - \alpha \nabla*\theta \mathcal{L}_i(\theta)
$$

### Outer Loop (Meta-Update)

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_i(\theta'_i)
$$

---

### Interpretation

* $\theta$ is a good initialization
* Few updates yield good task performance

---

## RL²

### Core Idea

Use a **recurrent policy** that implicitly learns the task.

* Policy input includes:
	* Observation
	* Reward
	* Previous action

The RNN hidden state stores task knowledge.

---

### Comparison

| Method | How it adapts    |
| ------ | ---------------- |
| MAML   | Gradient updates |
| RL²    | Hidden state     |

---

### Summary

Meta-RL enables agents to rapidly adapt to new tasks by learning either **initializations (MAML)** or **learning dynamics (RL²)** across task distributions.

---

## 5. Multi-Agent RL (MADDPG, QMIX)

### Motivation

Multiple agents introduce:

* Non-stationarity
* Coordination challenges

---

## MADDPG

### Centralized Training, Decentralized Execution (CTDE)

Each agent has:

* Actor: $\pi_i(a_i|o_i)$
* Centralized critic:
  $$
  Q_i(s, a_1, ..., a_N)
  $$

---

### Advantage

* Stable training
* Handles continuous actions

---

## QMIX

### Value Decomposition

Total value:
$$
Q_{tot} = f(Q_1, Q_2, ..., Q_N)
$$
With constraint:
$$
\frac{\partial Q_{tot}}{\partial Q_i} \ge 0
$$

This ensures:

* Greedy local actions are globally optimal

---

### Comparison

| Algorithm | Action Space | Approach            |
| --------- | ------------ | ------------------- |
| MADDPG    | Continuous   | Actor-Critic        |
| QMIX      | Discrete     | Value Decomposition |

---

### Summary

Multi-Agent RL methods address coordination via **centralized training** or **value decomposition**, enabling scalable learning in cooperative and competitive environments.

---

## 6. Offline RL (CQL – Conservative Q-Learning)

### Motivation

Offline data:

* No environment interaction
* Distribution shift risk

---

### Problem

Standard Q-learning **overestimates unseen actions**.

---

### Conservative Q-Learning (CQL)

Penalizes Q-values of unseen actions:
$$
\mathcal{L}*{CQL} = \mathbb{E}*{s}\left[
\log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim D}[Q(s,a)]
\right]
$$

---

### Effect

* Pushes down Q-values for OOD actions
* Prefers dataset-supported actions

---

### Use Cases

* Robotics logs
* Medical decision making
* Autonomous driving

---

### Summary

Offline RL focuses on learning from **fixed datasets**, and CQL ensures safety by **conservatively estimating Q-values**, preventing over-optimistic policies.

---

## 7. Imitation Learning (BC, GAIL)

---

### Behavioral Cloning (BC)

#### Idea

Supervised learning:
$$
\mathcal{L} = \mathbb{E}[|a - \pi(s)|^2]
$$

---

#### Limitations

* Covariate shift
* Errors compound over time

---

### GAIL (Generative Adversarial Imitation Learning)

#### Setup

* Generator: policy $\pi$
* Discriminator: $D(s,a)$

---

#### Objective

$$
\min_\pi \max_D ;
\mathbb{E}*{\pi}[\log D(s,a)] +
\mathbb{E}*{\pi_E}[\log(1 - D(s,a))]
$$

---

#### Interpretation

* Match expert occupancy measure
* No explicit reward required

---

### Summary

Imitation learning allows agents to learn from **demonstrations**, with BC being simple but brittle, and GAIL providing robustness through adversarial training.

---

## 8. RLHF (Reinforcement Learning from Human Feedback)

### Motivation

True rewards are:

* Hard to specify
* Misaligned with human intent

---

### RLHF Pipeline

1. **Supervised Fine-Tuning**
   Train policy on human demonstrations

2. **Reward Model Training**
   Humans rank outputs:
   $$
   \mathcal{L}_{RM} = -\log \sigma(r(x,y^+) - r(x,y^-))
   $$

3. **RL Optimization**
   Optimize policy using PPO:
   $$
   \max_\pi \mathbb{E}[r_\phi(x,y)] - \beta \text{KL}(\pi || \pi_{ref})
   $$

---

### Key Challenges

* Reward hacking
* Scalability of human feedback
* Alignment stability

---

### Summary

RLHF aligns policies with human preferences by learning a reward model from human feedback and optimizing behavior through constrained reinforcement learning.

---

## Final High-Level Summary

Advanced RL methods extend beyond standard model-free learning by:

* Using **models** (Dyna-Q, Dreamer, MuZero)
* Exploiting **structure** (Hierarchical RL)
* Enabling **fast adaptation** (Meta-RL)
* Handling **multiple agents**
* Learning from **fixed datasets**
* Learning from **demonstrations**
* Aligning with **human values (RLHF)**

Together, these methods form the foundation of **modern scalable and real-world RL systems**.
