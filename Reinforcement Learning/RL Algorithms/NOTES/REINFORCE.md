### Sources:
- [REINFORCE: Reinforcement Learning Most Fundamental Algorithm](https://www.youtube.com/watch?v=5eSh5F8gjWU)
- [REINFORCE (Vanilla Policy Gradient VPG) Algorithm Explained | Deep Reinforcement Learning](https://www.youtube.com/watch?v=boEO7tN7uoY)

## The REINFORCE Algorithm (Monte Carlo Policy Gradient)

The REINFORCE algorithm is a foundational method in **Policy-Based Reinforcement Learning**. It addresses the challenge of *directly* optimizing a policy function without relying on value estimates, marking a significant shift from Q-learning and DQN.

### The Non-Differentiable Environment Problem

In Q-learning and DQN, you train a function (the Q-function) to predict a target value, making it a standard supervised regression problem. REINFORCE, however, aims to directly train a **Policy Network** ($\pi_\theta$) whose output is a probability distribution over actions, $\pi_\theta(a|s)$.

* **Policy Network ($\pi_\theta$):** A neural network that takes the state ($s$) and outputs the probability of taking each action ($a$).
* **The Problem:** We want to find the gradient of the **expected total reward** with respect to the network's parameters ($\theta$). However, the path from the parameters ($\theta$) to the final reward ($R$) goes through the **environment** and the **stochastic action sampling**.
* **The Consequence:** Because the environment's dynamics (state transitions and rewards) are typically *unknown* and often *non-deterministic*, they are **not a fixed, differentiable function**. We **cannot compute the derivative** across this environment boundary to **backpropagate** the reward signal directly.

This is the core challenge: $\nabla_\theta E[\text{Total Reward}]$ is **hard** to compute.

> **Summary:** We cannot use standard backpropagation to optimize the Policy Network because the path from the network parameters to the final reward is broken by the **non-differentiable environment** and the **stochastic sampling of actions**.

### The Policy Network and Stochastic Sampling

To implement the policy, we need a function that outputs **probabilities** and allows for sampling based on those probabilities.

* **Pytorch Implementation:** `torch.distributions.Categorical(probs)`
    * The Policy Network outputs a tensor of probabilities (`probs`) for each action.
    * This tensor is wrapped into a `Categorical` distribution object.

* **Action Selection (Exploration):** `Dist.sample()`
    * The agent samples an action from this distribution based on the calculated probabilities. This ensures inherent **exploration** (unlike Q-learning, where exploration is forced by $\epsilon$-greedy). The higher the probability for an action, the more likely it is to be selected.

* **The Key for Gradient Calculation:** `Dist.log_prob(action)`
    * The Policy Gradient Theorem provides a mathematically sound way to calculate the gradient **without differentiating** through the environment. It relies on a critical identity: the gradient of a probability $\nabla_\theta P$ is proportional to $P \cdot \nabla_\theta \log P$.

    * The REINFORCE algorithm uses the $\mathbf{\text{log\_probability}}$ of the action taken to calculate the gradient of the objective function.

> **Summary:** The Policy Network uses a **stochastic distribution** (like `Categorical` in Pytorch) to output action probabilities. The key to the update is using the **log-probability** ($\log \pi_\theta(a|s)$) of the selected action, which is the essential component for estimating the non-differentiable gradient.

### The REINFORCE Update Rule

REINFORCE is a **Monte Carlo** algorithm because it requires running a **full episode** to completion before calculating the total reward (Return, $G_t$).

> **Note:** Immediate rewards $R_t$ are the raw rewards received **at each time step**. Returns $G_t$ are the **discounted sum of future rewards** from time step $t$ onward:
>
> $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$$
>
> Returns incorporate both *immediate* and future rewards, capturing the long-term effect of each action.

#### 1. The Policy Gradient Theorem (REINFORCE Form)

The goal is to maximize the expected return $J(\theta) = E_{\pi_\theta}[G_0]$. The theorem states that the gradient of this objective is:

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Where:
* $G_t$: The **return** (total discounted reward) accumulated **from time step $t$ until the end of the episode**.
* $\nabla_\theta \log \pi_\theta(a_t|s_t)$: The gradient of the *log probability* of the action $a_t$ taken in state $s_t$.

#### 2. The Loss/Objective Function

In Pytorch, instead of explicitly calculating the gradient, we define a **surrogate loss function** $L(\theta)$ such that when we calculate its gradient (e.g., using `loss.backward()`), we get the Policy Gradient estimate.

For REINFORCE, the objective function we want to *maximize* is:
$$J(\theta) = \sum_{t=0}^{T-1} G_t \cdot \log \pi_\theta(a_t|s_t)$$

In practice, to use gradient *descent* (which *minimizes* loss), we define the **Loss Function** as the negative of the objective:
$$\mathcal{L}(\theta) = -\sum_{t=0}^{T-1} G_t \cdot \log \pi_\theta(a_t|s_t)$$

*Minimizing* this **loss** is equivalent to *maximizing* the **expected return**.

> **Summary:** REINFORCE is a **Monte Carlo** method that requires completing an episode to calculate the **Return ($G_t$)**. The loss function used for gradient descent is the **negative** of the sum of $G_t$ multiplied by the **log probability** of the actions taken throughout the episode.

---

## Numerical Example: The Gradient Calculation

Imagine a very short episode with $\gamma = 0.5$.

| Time $t$ | State $s_t$ | Action $a_t$ | Immediate Reward $R_{t+1}$ | $\log \pi_\theta(a_t\|s_t)$ |
| :------: | :---------: | :----------: | :------------------------: | :-------------------------: |
|    0     |    $S_0$    |    $A_1$     |             10             |            -1.5             |
|    1     |    $S_1$    |    $A_2$     |             2              |            -0.8             |
|    2     |    $S_2$    |    $A_3$     |      0 (Episode Ends)      |              -              |

### Step 1: Calculate the Return ($G_t$)

We calculate the total discounted return backwards:
* **$G_2$:** Reward from $t=2$ onwards (none) = 0
* **$G_1$:** $R_2 + \gamma G_2 = 2 + 0.5(0) = \mathbf{2.0}$
* **$G_0$:** $R_1 + \gamma G_1 = 10 + 0.5(2.0) = \mathbf{11.0}$

> **Note:** Here, $R_t$ is the immediate reward at each step, while $G_t$ is the discounted sum of future rewards. Returns give a better signal for learning which actions have long-term value.

### Step 2: Calculate the Objective/Loss Components

Now we calculate the individual terms needed for the objective function $J(\theta) = \sum G_t \cdot \log \pi_\theta(a_t|s_t)$.

| Time $t$ | $G_t$ | $\log \pi_\theta(a_t\|s_t)$ | $G_t \cdot \log \pi_\theta(a_t\|s_t)$ |
| :------: | :---: | :-------------------------: | :-----------------------------------: |
|    0     | 11.0  |            -1.5             |                 -16.5                 |
|    1     |  2.0  |            -0.8             |                 -1.6                  |

### Step 3: Calculate the Loss and Perform Update

1.  **Objective Value ($J(\theta)$):**
    $$J(\theta) = (-16.5) + (-1.6) = -18.1$$

2.  **REINFORCE Loss Function ($\mathcal{L}(\theta)$):**
    $$\mathcal{L}(\theta) = -J(\theta) = -(-18.1) = \mathbf{18.1}$$

3.  **Update:** The agent calls `loss.backward()`.
    * The Pytorch framework uses the chain rule to compute the gradient $\nabla_\theta \mathcal{L}(\theta)$.
    * Crucially, the gradient of the $\log \pi_\theta(a_t|s_t)$ term is computed, while the $G_t$ term is treated as a **constant weight** (or **baseline**) since it is calculated from the non-differentiable environment.
    * An optimizer (e.g., Adam) then updates the Policy Network's weights ($\theta$) to **reduce the loss** of 18.1.

*Since the return $G_t$ for $t=0$ was large (11.0) and the $\log$ probability for $A_1$ was relatively small (-1.5), the resulting negative term (-16.5) contributes most of the loss. By minimizing the loss, the network will be driven to **increase the probability** of taking action $A_1$ in state $S_0$ in future episodes.*
