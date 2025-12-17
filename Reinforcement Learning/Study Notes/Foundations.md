## 1. Markov Decision Processes (MDPs)

**Definition:**
A **Markov Decision Process (MDP)** is a mathematical framework for modeling decision-making problems where outcomes are partly random and partly under the control of an agent. MDPs form the foundation of most RL algorithms.

**Components:**

An MDP is defined as a tuple:
$$ (\mathcal{S}, \mathcal{A}, P, R, \gamma) $$

1. **States ($\mathcal{S}$)** – All possible states the agent can be in.  
2. **Actions ($\mathcal{A}$)** – Set of all possible actions the agent can take.  
3. **Transition Probability ($P(s' \mid s,a)$)** – Probability of moving to state $s'$ given current state $s$ and action $a$.  
4. **Reward Function ($R(s,a,s')$)** – Expected immediate reward received after taking action $a$ in state $s$ and transitioning to $s'$.  
5. **Discount Factor ($\gamma \in [0,1]$)** – Determines the importance of future rewards.

**Markov Property:**

* The next state depends **only on the current state and action**, not on past states:
  $$ P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t) $$

**Goal of RL:**

* Find a **policy** $\pi(a\mid s)$ that maximizes the **expected cumulative discounted reward**:
  $$ G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} $$

---

## 2. Bellman Equations

Bellman equations describe **recursive relationships** for values in an MDP.

### 2.1 Value Function

The **state-value function** $V^\pi(s)$ under policy $\pi$ is the expected return starting from $s$ and following $\pi$:
$$ V^\pi(s) = \mathbb{E}_\pi\!\left[ \sum_{t=0}^{\infty} \gamma^t r_t \;\middle\vert\; s_0 = s \right] $$

**Bellman Expectation Equation (Value function):**
$$ V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a) \big[ R(s,a,s') + \gamma V^\pi(s') \big] $$

### 2.2 Action-Value Function (Q-function)

The **Q-function** $Q^\pi(s,a)$ is the expected return after taking action $a$ in state $s$ and following $\pi$ afterward:
$$ Q^\pi(s,a) = \mathbb{E}_\pi\!\left[ r_0 + \gamma V^\pi(s_1) \;\middle\vert\; s_0 = s, a_0 = a \right] $$

**Bellman Expectation Equation (Q-function):**
$$ Q^\pi(s,a) = \sum_{s'} P(s'\mid s,a) \Big[ R(s,a,s') + \gamma \sum_{a'} \pi(a'\mid s')\, Q^\pi(s',a') \Big] $$

### 2.3 Advantage Function

The **Advantage function** measures how much better taking action $a$ is compared to the average under policy $\pi$:
$$ A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) $$

* Used in **policy-gradient methods** (e.g., A2C, PPO) to reduce variance in updates.

---

## 3. Exploration vs. Exploitation

* **Exploitation:** Choose the action with the highest estimated value (greedy).  
* **Exploration:** Try other actions to discover potentially better rewards.

**Trade-off:** RL agents must balance **learning new knowledge** and **maximizing reward**.

### Common Exploration Strategies:

1. **ε-greedy policy**

* With probability $\epsilon$, pick a random action (explore).  
* With probability $1-\epsilon$, pick the best action (exploit).  
* Simple, widely used in discrete-action RL (e.g., DQN).

2. **Softmax / Boltzmann policy**

* Converts Q-values to probabilities:
  $$ \pi(a\mid s) = \frac{e^{Q(s,a)/\tau}}{\sum_b e^{Q(s,b)/\tau}} $$
* $\tau$ = temperature: high $\tau$ → more exploration, low $\tau$ → more exploitation.

3. **Noise in continuous actions**

* In DDPG: add **OU noise** to deterministic actions for exploration.  
* In SAC: stochastic policies naturally provide exploration.

---

## 4. Policy Evaluation & Policy Improvement

### 4.1 Policy Evaluation

* Compute $V^\pi(s)$ or $Q^\pi(s,a)$ **for a fixed policy**.  
* Solve **Bellman expectation equation** iteratively (or approximately with function approximators).

### 4.2 Policy Improvement

* Generate a new policy $\pi'$ that is **greedy w.r.t. the current value function**:
  $$ \pi'(s) = \arg\max_a Q^\pi(s,a) $$
* This guarantees $V^{\pi'}(s) \ge V^\pi(s)$ (Policy Improvement Theorem).

**Policy Iteration:**

* Alternate between **policy evaluation** and **policy improvement** until convergence.

---

## 5. Value Iteration / Policy Iteration

### 5.1 Policy Iteration

1. Initialize policy $\pi$.  
2. **Policy evaluation:** compute $V^\pi(s)$ for all $s$.  
3. **Policy improvement:** update $\pi$ to be greedy w.r.t.\ $V^\pi$.  
4. Repeat until policy converges.

### 5.2 Value Iteration

* Combines evaluation and improvement in a **single step** using the **Bellman optimality equation**:
  $$ V^*(s) = \max_a \sum_{s'} P(s'\mid s,a) \big[ R(s,a,s') + \gamma V^*(s') \big] $$
* Iterate until convergence.  
* Optimal policy is:
  $$ \pi^*(s) = \arg\max_a \sum_{s'} P(s'\mid s,a) \big[ R(s,a,s') + \gamma V^*(s') \big] $$

**Key difference:**

* **Policy iteration:** separate evaluation and improvement.  
* **Value iteration:** combine evaluation and improvement in one update.

---
