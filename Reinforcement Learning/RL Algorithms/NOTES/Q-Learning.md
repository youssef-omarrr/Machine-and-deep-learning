### Sources
- [Q-learning - Explained!](https://www.youtube.com/watch?v=TiAXhVAZQl8)
- [Reinforcement Learning: on-policy vs off-policy algorithms](https://www.youtube.com/watch?v=YUKUXoUg3Nc)
# Q-Learning notes

![](../../imgs/PastedImage.png)

### Q-table

The **Q-table** (or Q-function, $Q(s, a)$) is the central element in Q-learning. It is a lookup table where each entry stores the **quality** (Q-value) of taking a specific **action** ($a$) while being in a specific **state** ($s$).

| State |     $A_1$     |     $A_2$     |
| :---: | :-----------: | :-----------: |
| $S_1$ | $Q(s_1, A_1)$ | $Q(s_1, A_2)$ |
| $S_2$ | $Q(s_2, A_1)$ | $Q(s_2, A_2)$ |

* **State ($s$):** The current situation or configuration of the environment.
* **Action ($a$):** A move the agent can make from the current state.
* **Q-Value ($Q(s, a)$):** The **expected maximum future reward** the agent will receive starting from state $s$, taking action $a$, and then following the optimal policy thereafter.

The agent's goal is to learn the **optimal Q-table** (often denoted $Q^*$). Once $Q^*$ is learned, the **optimal policy** $\pi^*$ is simply to choose the action $a$ in state $s$ that has the highest Q-value:
$$\pi^*(s) = \underset{a}{\operatorname{argmax}} Q^*(s, a)$$

> **Summary:** The **Q-table** is a matrix that stores the expected future reward for every **state-action pair** ($Q(s, a)$). The agent uses this table to decide the **best action** to take by choosing the action with the highest Q-value.

### Bellman Equation and Discount Factor ($\gamma$)

The **Bellman Equation** forms the mathematical foundation for updating the Q-values. Q-learning specifically uses the **Optimal Bellman Equation** for Q-functions:

$$Q(s, a) = R_{t+1} + \gamma \max_{a'} Q(s', a')$$

This equation states that the Q-value for the *current* state-action pair ($s, a$) is equal to the **immediate reward** ($R_{t+1}$) received from taking action $a$, plus the **discounted maximum future Q-value** from the *next state* ($s'$).

* **$R_{t+1}$:** The **immediate reward received** after moving from $s$ to $s'$ by taking action $a$.
* **$\gamma$ (Gamma, Discount Factor):** A value between $0$ and $1$ ($0 \le \gamma \le 1$) that determines the importance of **future rewards**.
    * If $\gamma$ is close to **0**, the agent is **myopic** (short-sighted) and only cares about *immediate* rewards.
    * If $\gamma$ is close to **1**, the agent is **far-sighted** and values *future* rewards almost as much as immediate ones.

> **Summary:** The **Bellman Equation** relates the current Q-value to the immediate reward and the **maximum Q-value** of the **next state**. The **discount factor $\gamma$** weighs how much the agent values immediate rewards versus future rewards.

### TD Error

The **Temporal Difference (TD) Error** is the difference between the agent's **current estimate** of a Q-value and a **more informed (updated) estimate** based on the reward received and the Q-value of the next state.

$$\text{TD Error} = (\text{New Estimate}) - (\text{Old Estimate})$$

The new estimate, called the **TD Target**, is derived from the Bellman equation:
$$\text{TD Target} = R_{t+1} + \gamma \max_{a'} Q(s', a')$$

Therefore:
$$\text{TD Error} = \left[ R_{t+1} + \gamma \max_{a'} Q(s', a') \right] - Q(s, a)$$

The TD error quantifies **how wrong** the agent's current prediction $Q(s, a)$ is. The goal of the learning process is to drive this error towards *zero* by adjusting $Q(s, a)$.

> **Summary:** The **TD Error** is the difference between the **current Q-value estimate** and the **TD Target** (a better estimate incorporating the actual reward and the next state's value). It measures the error in the agent's prediction.

### Update Rule and Learning Rate ($\alpha$)

The Q-learning **Update Rule** uses the TD Error to incrementally **adjust the current Q-value** towards the TD Target:

$$\text{New } Q(s, a) = Q(s, a) + \alpha \times [\text{TD Error}]$$

Substituting the TD Error expression:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

* **$\alpha$ (Alpha, Learning Rate):** A value between $0$ and $1$ ($0 < \alpha \le 1$) that controls **how much** of the TD Error is incorporated into the new Q-value.
    * A high $\alpha$ (e.g., $\alpha=1$) means the agent **fully accepts** the new estimate, potentially leading to faster but unstable learning.
    * A low $\alpha$ means the agent makes **small, cautious updates**, leading to slower but more stable convergence.

> **Summary:** The **Update Rule** is an iterative process where the Q-value $Q(s, a)$ is adjusted using the **TD Error** multiplied by the **learning rate $\alpha$**. $\alpha$ controls the magnitude of the update.

---

## Policy Distinction: On-Policy vs. Off-Policy

### Target Policy ($\pi_T$) vs. Behavior Policy ($\pi_B$)

* **Target Policy ($\pi_T$):** This is the **optimal policy** the agent is ultimately trying to **learn**. It is usually a **greedy** policy that exploits the current knowledge by choosing the action with the **highest** Q-value: $\pi_T(s) = \underset{a}{\operatorname{argmax}} Q(s, a)$.
* **Behavior Policy ($\pi_B$):** This is the policy the agent uses to **explore** the environment and **collect data** (experiences). To ensure the agent explores and finds the true optimal path, $\pi_B$ is typically **stochastic** (e.g., $\epsilon$-greedy, where it acts greedily *most of the time* but takes a **random** action with probability $\epsilon$).

### On-Policy vs. Off-Policy

This is the key distinction in value-based RL algorithms:

* **On-Policy ($\pi_B$ must be the same as $\pi_T$):** The agent learns the value of the policy ($\pi_B$) it is currently using to interact with the environment.
    * *Example:* **SARSA** (State-Action-Reward-State-Action). The update uses the Q-value of the *actual* action taken in the next state: $Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma Q(\mathbf{s', a'}) - Q(s, a) \right]$. If $\pi_B$ is $\epsilon$-greedy, SARSA learns the optimal $\epsilon$-greedy policy.
	> Note how there is no $\max$ term in this equation.
	
* **Off-Policy ($\pi_B$ is different from $\pi_T$):** The agent learns the value of the **optimal policy ($\pi_T$)** while following a **different, exploratory policy ($\pi_B$)** to collect data.

> **Summary:** The **Target Policy ($\pi_T$)** is the policy we *want* to learn (usually greedy). The **Behavior Policy ($\pi_B$)** is the policy used to *collect data* (usually $\epsilon$-greedy). 
> **On-policy** methods like SARSA learn the value of the policy being executed ($\pi_B$). 
> **Off-policy** methods like Q-learning learn the value of the optimal policy ($\pi_T$) regardless of the policy being executed ($\pi_B$).

### Q-Learning is Off-Policy

Q-learning is an **off-policy** algorithm because:

1.  **Behavior Policy ($\pi_B$):** It uses a data collection policy like $\epsilon$-greedy (or even random) to ensure **exploration** and gather diverse experiences.
2.  **Target Policy ($\pi_T$):** The update rule *always* uses the **greedy** choice for the next state's value, regardless of which action was *actually* taken by $\pi_B$. This is seen in the use of the **$\max$ operator** in the update: $\gamma \mathbf{\max_{a'} Q(s', a')}$. This $\max$ term represents the value of the optimal, greedy target policy $\pi_T$.

By **decoupling** the data collection (exploration) from the value estimation (exploitation), Q-learning is guaranteed to converge to the **optimal Q-function ($Q^*$)**, even if the agent is constantly exploring via $\pi_B$.

> **Summary:** **Q-learning is Off-Policy**. It uses a flexible, exploratory **Behavior Policy** (like $\epsilon$-greedy) to move through the environment, but its update rule (using the $\max$ function) focuses on learning the value of the optimal, **Greedy Target Policy**.

---

## Numerical Example of Q-Table Update

Let's consider a simple scenario with two states, $S_1$ and $S_2$, and two actions, $A_L$ (Left) and $A_R$ (Right).

* **Parameters:**
    * Learning Rate ($\alpha$) = $\mathbf{0.5}$
    * Discount Factor ($\gamma$) = $\mathbf{0.9}$

### Initial Q-Table

Assume the Q-table is initialized with zeros (or small random numbers):

| State | $A_L$ | $A_R$ |
| :---: | :---: | :---: |
| $S_1$ | 0.00  | 0.00  |
| $S_2$ | 0.00  | 0.00  |

### Episode Transition (One Step)

The agent starts in $S_1$ and, following its **Behavior Policy** ($\pi_B$, e.g., $\epsilon$-greedy), decides to take $A_R$.

* **Transition:** Agent moves from $\mathbf{s = S_1}$ by taking $\mathbf{a = A_R}$.
* **Outcome:** It lands in $\mathbf{s' = S_2}$ and receives a $\mathbf{Reward (R_{t+1}) = 10}$.

### Q-Learning Update Calculation

We need to update $Q(s, a) = Q(S_1, A_R)$ using the update rule:
$$Q(S_1, A_R) \leftarrow Q(S_1, A_R) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s', a') - Q(S_1, A_R) \right]$$

1.  **Current Q-Value (Old Estimate):**
    $$Q(S_1, A_R) = 0.00$$

2.  **Maximum Future Value:**
    Since the next state is $S_2$, we look at the values in row $S_2$ of the current Q-table and find the maximum:
    $$\max_{a'} Q(S_2, a') = \max(Q(S_2, A_L), Q(S_2, A_R)) = \max(0.00, 0.00) = \mathbf{0.00}$$

3.  **TD Target (New Estimate):**
    $$\text{TD Target} = R_{t+1} + \gamma \max_{a'} Q(s', a')$$
    $$\text{TD Target} = 10 + 0.9 \times 0.00 = \mathbf{10.00}$$

4.  **TD Error:**
    $$\text{TD Error} = \text{TD Target} - \text{Old Estimate}$$
    $$\text{TD Error} = 10.00 - 0.00 = \mathbf{10.00}$$

5.  **New Q-Value (Update):**
    $$\text{New } Q(S_1, A_R) = Q(S_1, A_R) + \alpha \times \text{TD Error}$$
    $$\text{New } Q(S_1, A_R) = 0.00 + 0.5 \times 10.00 = \mathbf{5.00}$$

### Updated Q-Table

The Q-table after this single transition:

| State | $A_L$ | $A_R$ |
| :---: | :---: | :---: |
| $S_1$ | 0.00 | **5.00** |
| $S_2$ | 0.00 | 0.00 |

The Q-value $Q(S_1, A_R)$ has been updated to reflect the reward experienced. As the agent continues to explore, these values will iteratively **converge to the optimal values**.