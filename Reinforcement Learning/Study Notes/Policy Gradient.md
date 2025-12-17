A **policy gradient** is a type of reinforcement learning method that **directly optimizes the policy** instead of learning a value function first. Let’s break it down as simply and clearly as possible.

---

## 1. Core Idea

* Traditional methods (like Q-learning) focus on **learning a value function** $Q(s,a)$ and then deriving a policy (e.g., greedy).
* **Policy gradient methods** instead **parameterize the policy** $\pi_\theta(a|s)$ directly with parameters $\theta$ (usually a neural network).
* The goal is to **maximize expected cumulative reward**:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \sum_{t=0}^{T} \gamma^t r_t \Big]
$$

Where \( $\tau$ \) is a trajectory (sequence of states and actions) sampled from the policy.

---

## 2. How It Works

1. **Define a parameterized policy** \( $\pi_\theta(a|s$) \)

   * Continuous or discrete actions.
   * Could output probabilities (discrete) or distribution parameters (continuous, e.g., mean/std).

2. **Compute gradient of expected reward** with respect to policy parameters:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t \Big]
$$

   * This is called the **REINFORCE rule** (Williams, 1992).
   * $G_t$ = cumulative reward from time $t$.
   * Intuition: increase probability of actions that lead to high rewards.

3. **Update parameters using gradient ascent:**
$$
\theta \gets \theta + \alpha \nabla_\theta J(\theta)
$$

---

## 3. Key Concepts

* **Stochastic Policy:** Policy must be **probabilistic** to allow differentiation through sampling.
* **Baseline / Advantage Function:**

  * Direct use of ($G_t$) has high variance.
  * Subtract a baseline (like $V(s_t)$) to reduce variance:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ \nabla_\theta \log \pi_\theta(a_t|s_t)\, (G_t - b(s_t)) \big]
$$
  * Common choice: \($b(s_t) = V^\pi(s_t)$\).
* **Actor-Critic Methods:**

  * **Actor** = policy network ($\pi_\theta$)
  * **Critic** = value network ($V_\phi(s)$ ) used as baseline.
  * Reduces variance and stabilizes learning (used in A2C, PPO, SAC).

---

## 4. Pros and Cons

| Pros                                                                | Cons                                                   |
| ------------------------------------------------------------------- | ------------------------------------------------------ |
| Can handle **continuous and high-dimensional action spaces** easily | High variance, slow convergence                        |
| Directly optimizes **stochastic policies**                          | Requires careful tuning of learning rate and baselines |
| Naturally handles **on-policy learning**                            | On-policy methods need fresh samples every update      |
| Works well with **neural network function approximation**           | Sensitive to reward scaling                            |

---

### 5. Quick Intuition

1. Imagine the policy as a “probability knob” for each action.
2. When an action leads to high reward, **turn the knob up** for that action in that state.
3. When an action leads to low reward, **turn the knob down**.
4. Repeat over many episodes → policy learns to favor **high-reward actions**.

---
**policy gradient methods are generally model-free algorithms**, not model-based. 
## 1. Model-Free vs Model-Based RL

| Type            | What it Does                                                                                                                                      | Example Methods                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Model-Free**  | Learns **directly from interaction with the environment**, without trying to model the transition dynamics ($P(s' \mid s,a)$ or rewards $R(s,a)$. | Q-Learning, DQN, SARSA, Policy Gradients (REINFORCE, PPO, SAC) |
| **Model-Based** | Learns or uses a **model of the environment** (transition & reward functions) and plans ahead using this model.                                   | Dyna-Q, MuZero, MBPO, MPC (Model Predictive Control)           |

---

## 2. Why Policy Gradient is Model-Free

* Policy gradients **do not need to know ($P(s'|s,a)$)**.
* They learn **a mapping from states to actions** by **sampling trajectories from the environment** and **estimating returns**.
* Updates are based on **observed rewards**, not on predicted next states.

### Example:

* REINFORCE updates:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t \Big]
$$
* Here, \( $\tau$ \) comes from **actual interaction** with the environment, no model of $P$ is required.

---

**Summary:**

* Policy gradients = **model-free, on-policy methods**.
* You can combine them with a learned model to make them model-based (rarely done in practice), but standard PG algorithms like REINFORCE, A2C, PPO, SAC are all **model-free**.

---
