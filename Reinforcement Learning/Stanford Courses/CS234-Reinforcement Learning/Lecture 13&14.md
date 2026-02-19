# Lecture 13
* **Fast RL in MDP**

  * Goal: achieve *sample-efficient* learning in finite MDPs with state space $\mathcal{S}$, action space $\mathcal{A}$, horizon $H$ or discount $\gamma$.
  * An MDP is defined as $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.
  * Objective:
    $$
    V^\pi(s) = \mathbb{E}*\pi \left[ \sum*{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]
    $$
  * *Fast RL* typically means achieving **polynomial sample complexity** in $|\mathcal{S}|, |\mathcal{A}|, \frac{1}{1-\gamma}, \frac{1}{\epsilon}, \log\frac{1}{\delta}$.

* **Model-based interval estimation with exploration bonus (MBIE-EB) → Upper Confidence Bound idea in MDPs**

  * Extension of UCB from bandits to MDPs.
  * Maintain empirical transition $\hat P(s'|s,a)$ and reward $\hat R(s,a)$.
  * Add *exploration bonus*:
    $$
    b(s,a) = \beta \sqrt{\frac{1}{N(s,a)}}
    $$
  * Optimistic Q-value:

- Encourages visiting *under-explored* $(s,a)$ pairs.

- This is the **UCB principle generalized to MDPs**.

- **PAC for MDPs**

  * PAC-MDP: algorithm is *Probably Approximately Correct* if:

    * With probability $\ge 1-\delta$
    * It behaves $\epsilon$-optimally
    * Except for at most polynomially many steps.
  * Formal:
    $$
    V^\pi(s) \ge V^*(s) - \epsilon
    $$
  * MBIE-EB is PAC-MDP.

- **Simulation lemma**

  * Quantifies how model errors affect value estimates.
  * If:
    $$
    |P - \hat P|_1 \le \epsilon_P, \quad |R - \hat R| \le \epsilon_R
    $$
  * Then:
    $$
    |V^\pi - \hat V^\pi|*\infty \le \frac{\epsilon_R}{1-\gamma} + \frac{\gamma \epsilon_P R*{\max}}{(1-\gamma)^2}
    $$
  * Important: **small model errors amplify by $\frac{1}{(1-\gamma)^2}$**.

- **Bayesian MDP**

  * Treat $P, R$ as random variables with prior:
    $$
    P, R \sim p(\theta)
    $$
  * Maintain posterior:
    $$
    p(\theta | \mathcal{D})
    $$
  * Optimal solution is belief-MDP (intractable in general).

- **Thompson sampling for bandits**

  * Posterior over reward parameters.
  * Algorithm:

    * Sample $\theta \sim p(\theta|\mathcal{D})$
    * Play $a = \arg\max_a \mathbb{E}[r|a,\theta]$
  * Naturally balances exploration via posterior uncertainty.

- **Bayesian model-based RL**

  * Maintain posterior over transition and reward models.
  * Planning integrates over posterior:
    $$
    V(s) = \mathbb{E}*{\theta \sim p(\theta|\mathcal{D})}[V^\pi*\theta(s)]
    $$
  * Exact solution is computationally hard.

- **Thompson sampling model-based RL**

  * Sample one MDP:
    $$
    \theta \sim p(\theta|\mathcal{D})
    $$
  * Solve sampled MDP optimally.
  * Execute policy for some duration.

- **Posterior Sampling for RL (PSRL) → avoids thrashing**

  * Algorithm (episodic setting):

    ```cpp
    For episode k:
        Sample MDP Mk ~ posterior
        Compute optimal policy πk for Mk
        Execute πk for entire episode
        Update posterior
    ```

  * *Avoids thrashing* because policy is fixed during episode.

  * Provides near-optimal Bayesian regret bounds.

- **Seed sampling and concurrent PSRL**

  * Instead of sampling full MDP each episode, sample a random seed that deterministically generates model samples.
  * Allows parallelization and consistent updates.
  * Useful in large-scale settings.

- **Seed sampling vs Thompson sampling**

  * Thompson sampling: re-samples model frequently.
  * Seed sampling: reuses randomness for stability.
  * Reduces oscillatory behavior.

- **Generalization and exploration**

  * Generalization reduces sample complexity.
  * Instead of learning each $(s,a)$ independently:
    $$
    Q(s,a) = \phi(s,a)^T w
    $$
  * Exploration must account for uncertainty in $w$.

- **Contextual multi-arm bandits → benefits of generalization**

  * Context $x_t$ observed before action.
  * Reward:
    $$
    r_t = f(x_t, a_t) + \epsilon
    $$
  * Enables transfer across contexts.

- **Disjoint linear contextual multi-arm bandits**

  * Separate parameter per arm:
    $$
    r = x^T \theta_a + \epsilon
    $$
  * Learn $\theta_a$ independently.

- **Learning in linear contextual bandits**

  * Ridge regression estimate:
    $$
    \hat \theta_a = A_a^{-1} b_a
    $$
    where:
    $$
    A_a = \sum x x^T + \lambda I, \quad b_a = \sum r x
    $$
  * LinUCB bonus:
    $$
    a = \arg\max_a \left( x^T \hat\theta_a + \alpha \sqrt{x^T A_a^{-1} x} \right)
    $$

- **Generalization and optimism**

  * Optimism in face of uncertainty:
    $$
    \hat r + \text{confidence radius}
    $$
  * Drives efficient exploration in linear settings.

---
# Lecture 14
* **Simulation-based search**

  * Use generative model to simulate rollouts.
  * Estimate:
    $$
    Q(s,a) \approx \frac{1}{N} \sum_i G_i
    $$

* **Computing action for current state only**

  * Online planning.
  * Focus on $s_{current}$ instead of full policy.

* **Simple Monte Carlo search**

  * For each action:

    * Simulate $N$ rollouts.
    * Average return.
  * High variance.

* **Forward search expectimax tree**

  * Build depth-$d$ tree.
  * Value:

- Disadvantage: tree size grows exponentially:
  $$
  O(|\mathcal{A}|^d |\mathcal{S}|^d)
  $$

- **Monte Carlo Tree Search (MCTS)**

  * Combines tree expansion + rollouts.
  * Four steps:

    1. Selection
    2. Expansion
    3. Simulation
    4. Backpropagation

- **Upper Confidence Tree (UCT)**

  * Applies UCB at each node:
    $$
    a = \arg\max_a \left( Q(s,a) + c \sqrt{\frac{\log N(s)}{N(s,a)}} \right)
    $$

  * Expands promising branches.

  * Scales to large action spaces.

  * Pseudocode:

    ```cpp
    function UCT(root):
        while budget not exhausted:
            node = select(root)
            child = expand(node)
            reward = simulate(child)
            backpropagate(child, reward)
        return argmax_a Q(root, a)
    ```

  * Balances exploration/exploitation locally.

- **Advantages of MCTS**

  * Avoids full tree expansion.
  * Anytime algorithm.
  * Works with black-box simulators.

- **AlphaGo and AlphaZero**

  * Use MCTS + neural networks.
  * Self-play: generate training data by playing against itself.
  * Curriculum emerges automatically from increasing opponent strength.

- **Train neural network to predict policies and values**

  * Policy head: $\pi_\theta(a|s)$
  * Value head: $v_\theta(s)$
  * Loss:
    $$
    \mathcal{L} = (z - v_\theta(s))^2 - \pi^T \log p_\theta + c|\theta|^2
    $$
  * MCTS uses network for prior + value estimates.
  * Iterative improvement via policy iteration with search.
