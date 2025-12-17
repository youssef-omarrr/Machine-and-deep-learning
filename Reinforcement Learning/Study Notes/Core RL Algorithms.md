## 1. Monte Carlo Methods

**Monte Carlo (MC) methods** estimate value functions by **sampling complete episodes** from the environment.

### Key Ideas:

* Use **actual returns** ($G_t$) instead of bootstrapping.
* Only works for **episodic tasks** (where episodes terminate).
* Two types:

1. **First-visit MC:**

   * Only updates the value of a state the **first time it appears** in an episode.
   * Formula:
     $$V(s) \gets V(s) + \alpha \bigl(G_t - V(s)\bigr)$$
     Where \($G_t$\) is the cumulative reward after first visit.

2. **Every-visit MC:**

   * Updates **every occurrence** of the state in the episode.
   * Same update rule; just averages over all visits.

**Pros:** Simple, unbiased for episodic tasks.
**Cons:** Requires **full episode**, high variance, slow for long episodes.

---

## 2. Temporal Difference Learning (TD)

**TD learning** combines **Monte Carlo ideas** (learning from returns) with **dynamic programming** (bootstrapping).

### TD(0):

* Updates value function **after each step**, using **next state value**:
  $$V(s_t) \gets V(s_t) + \alpha \bigl[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\bigr]$$
* Bootstraps: uses $V(s_{t+1})$ instead of the full return \($G_t$\).

### TD(\lambda) (Eligibility Traces):

* Introduces **λ parameter** to mix n-step returns:
  $$V(s) \gets V(s) + \alpha \,\delta_t\, e(s)$$
* $e(s)$ = **eligibility trace**, decays over time.
* λ = 0 → TD(0), λ = 1 → Monte Carlo.
* Provides **bias-variance trade-off** between TD and MC.

**Pros:** Works online, faster than MC, doesn’t require full episodes.

---

## 3. SARSA / Q-Learning

**SARSA and Q-Learning** are **model-free, on-policy / off-policy algorithms** for control (learning Q-values).

### 3.1 SARSA (On-Policy)

* Updates Q-values using the **action actually taken** by current policy:
  $$Q(s_t,a_t) \gets Q(s_t,a_t) + \alpha \big[ r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) \big]$$
* **On-policy:** learns about the policy being followed (e.g., ε-greedy).

### 3.2 Q-Learning (Off-Policy)

* Updates using the **maximal action value** at next state (optimal action):
  $$Q(s_t,a_t) \gets Q(s_t,a_t) + \alpha \big[ r_{t+1} + \gamma \max_a Q(s_{t+1},a) - Q(s_t,a_t) \big]$$
* **Off-policy:** can learn optimal policy independently of exploration.

**Difference:** SARSA is safer (takes exploration into account), Q-Learning is more aggressive (always targets optimal).

---

## 4. Deep Q-Network (DQN)

**DQN** scales Q-Learning to **high-dimensional / continuous state spaces** using **neural networks**.

### Key Ideas:

* **Q-network** approximates Q-values: $Q(s,a;\theta)$.
* **Replay Buffer:** stores past transitions to **break correlations** and enable off-policy learning.
* **Target Network:** separate network \(Q'\) to compute target values:
  $$y = r + \gamma \max_a Q'(s',a';\theta^-)$$

**Training:**

* Minimize **MSE loss**:
  $$L(\theta) = \mathbb{E}[(y - Q(s,a;\theta))^2]$$

**Pros:** Works with images / complex observations.
**Cons:** Can overestimate Q-values; sensitive to hyperparameters.

---

## 5. Improvements to DQN

### 5.1 Double DQN

* Addresses **Q-value overestimation** in DQN.
* Use online network to select action, target network to evaluate:
  $$y = r + \gamma\, Q'\bigl(s',\,\arg\max_a Q(s',a;\theta),\,\theta^- \bigr)$$

### 5.2 Dueling DQN

* Splits Q-function into **state value** $V(s)$ and **advantage** $A(s,a)$:
  $$Q(s,a) = V(s) + \Big(A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \Big)$$
* Helps network learn **which states are valuable** even when action choice is not critical.

### 5.3 Prioritized Experience Replay

* Samples transitions with **probability proportional to TD-error**.
* Focuses learning on **more important / surprising transitions**.

---

### Summary Table

| Method             | Type       | Update Rule                                       | Key Feature                       | Notes                                    |
| ------------------ | ---------- | ------------------------------------------------- | --------------------------------- | ---------------------------------------- |
| Monte Carlo        | On-policy  | $V(s) \gets V(s) + \alpha\,($G_t$ - V(s))$            | Uses full episode returns         | High variance, episodic only             |
| TD(0)              | On-policy  | $V(s) \gets V(s) + \alpha\,[r+\gamma V(s')-V(s)]$   | Bootstraps, step-wise updates     | Online learning                          |
| TD(\lambda)        | On-policy  | Eligibility traces                                | Mix of MC & TD                    | Bias-variance trade-off                  |
| SARSA              | On-policy  | $Q(s,a) \gets Q(s,a) + \alpha\,[r+\gamma Q(s',a')-Q(s,a)]$       | Follows current policy            | Safer, considers exploration             |
| Q-Learning         | Off-policy | $Q(s,a) \gets Q(s,a) + \alpha\,[r+\gamma \max_a Q(s',a)-Q(s,a)]$ | Targets optimal policy            | Aggressive, faster convergence           |
| DQN                | Off-policy | NN approximates Q                                 | Replay buffer + target network    | Works with high-dimensional states       |
| Double DQN         | Off-policy | Reduce overestimation                             | Separate selection & evaluation   | Improves stability                       |
| Dueling DQN        | Off-policy | Split V and A                                     | Learn state importance            | Useful when actions have similar effects |
| Prioritized Replay | Off-policy | Weighted sampling                                 | Focus learning on key transitions | Speeds up convergence                    |

---

