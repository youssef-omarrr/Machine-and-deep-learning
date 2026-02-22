# Lecture 15

* **Value alignment (the psychological definition)**
  * In psychology, *value alignment* refers to the degree to which an agent’s goals, preferences, or internal value system match those of a human or society.
  * In AI, this becomes the problem of designing agents whose **objective functions** reflect *what humans actually care about*, not just what is easy to specify.
  * Formally, if the human has a true (latent) utility function $U^*(s)$ but the designer specifies reward $R(s,a)$, misalignment occurs when:
    $$
    R \neq U^*
    $$
  * The agent optimizes:
    $$
    \pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t,a_t)\right]
    $$
    but what we *actually* want is:
    $$
    \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t U^*(s_t)\right]
    $$
  * The core alignment problem: **we do not know $U^*$ explicitly**, and specifying $R$ incorrectly leads to unintended behavior.
  * This connects to:

    * *Reward misspecification*
    * *Reward hacking*
    * *Specification gaming*
  * In RL terms, alignment is about ensuring the learned policy $\pi$ optimizes the *intended objective*, not merely the *formal reward signal*.

* **Aligning to user intent (and how it can go south)**

  * User intent is often:

    * Underspecified
    * Context-dependent
    * Implicit rather than explicit
  * If we model user intent as a latent variable $\theta$, then:
    $$
    \theta \sim p(\theta), \quad \text{agent must infer } \theta \text{ from behavior}
    $$
  * Problems arise when:

    * The proxy reward differs from actual intent:
      $$
      R_{\text{proxy}} \neq U_{\text{true}}
      $$
    * The agent exploits loopholes in the reward function.
    * The agent over-optimizes imperfect metrics (Goodhart’s Law).
  * **How it can go wrong:**

    * *Reward hacking*: agent finds degenerate strategy maximizing reward without solving real task.
    * *Specification gaming*: satisfies literal objective but violates spirit of task.
    * *Over-optimization*: extreme policies that exploit edge cases in system dynamics.
  * Example formulation:

    * If we optimize:
      $$
      \max_\pi \mathbb{E}[R_{\text{proxy}}]
      $$
      but true objective is:
      $$
      \max_\pi \mathbb{E}[U_{\text{true}}]
      $$
      then increasing optimization power can *increase divergence* between outcomes.
  * This motivates:

    * Inverse Reinforcement Learning (IRL):
      $$
      \hat R = \arg\max_R p(\text{demonstrations} \mid R)
      $$
    * Cooperative IRL (CIRL): human and AI modeled as cooperative agents.
    * Preference learning and human-in-the-loop RL.
  * Central difficulty: **intent is not directly observable**, and stronger optimization amplifies small misspecifications.

# Lecture 16
* **PPO is used in off-policy, but the very first step is always on-policy**

  * **Correction:** PPO is fundamentally an *on-policy* algorithm.
  * PPO optimizes the clipped surrogate objective:
    $$
    L^{\text{CLIP}}(\theta) = \mathbb{E}*t \left[ \min \left( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
    $$
    where
    $$
    r_t(\theta) = \frac{\pi*\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
    $$
  * Data is collected using $\pi_{\theta_{\text{old}}}$ → this is *on-policy sampling*.
  * PPO may reuse data for multiple gradient steps, but the data is still generated on-policy.
  * Key tricks:

    * Clipped objective
    * GAE (Generalized Advantage Estimation):
      $$
      A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
      $$
    * Value function baseline
    * Entropy bonus

* **For the following apps, what are the RL algorithms, functions, and tricks used**

  * **DeepMind AlphaTensor → learning matrix multiplication algorithms**

    * Uses RL framed as a *single-player game*.
    * State: partial tensor decomposition.
    * Action: select next rank-1 tensor.
    * Reward: negative number of scalar multiplications.
    * Uses:

      * Policy/value neural networks
      * Monte Carlo Tree Search (MCTS)
      * Self-play style improvement
    * Objective:
      $$
      \min (\text{scalar multiplications})
      $$
    * Key trick: represent matrix multiplication as tensor factorization.

  * **Learning plasma control for fusion science**

    * Applied in tokamak control (e.g., TCV reactor).
    * Continuous control.
    * Likely algorithms:

      * Model-based RL
      * MPC + neural network policies
      * Policy gradient methods (e.g., PPO)
    * Reward encodes:

      * Stability
      * Shape control
      * Safety constraints
    * Heavy use of:

      * Simulators
      * Domain randomization
      * Safety constraints
      * Offline pretraining before deployment

* **State dependence**

  * Bandits:

    * No state transition.
    * Reward:
      $$
      r_t \sim p(r|a_t)
      $$
    * No temporal dependence.
  * General decision process (MDP):

    * State evolves:
      $$
      s_{t+1} \sim P(\cdot|s_t,a_t)
      $$
    * Objective depends on long-term effects.
  * Key difference: *credit assignment across time*.

* **Online vs offline**

  * Online RL:

    * Agent collects data while learning.
    * Distribution shifts with policy.
  * Offline RL:

    * Fixed dataset $\mathcal{D}$.
    * No environment interaction.
    * Core issue: distributional shift.
      $$
      \pi(a|s) \text{ may select actions not in } \mathcal{D}
      $$

* **Function approximation + off-policy learning is a key challenge**

  * Known as the *deadly triad*:

    * Function approximation
    * Bootstrapping
    * Off-policy data

  * Can cause divergence.

  * **PPO**

    * On-policy.
    * Avoids instability of off-policy bootstrapping.

  * **DAgger (Dataset Aggregation)**

    * Imitation learning algorithm.
    * Algorithm:

      ```cpp
      Initialize D with expert demonstrations
      for iteration i:
          Train policy π_i on D
          Execute π_i to collect states
          Query expert for actions
          Add (state, expert_action) to D
      ```
    * Reduces covariate shift.

  * **Pessimistic Q-learning / CQL / MOPO**

    * Conservative Q-Learning (CQL):

      * Penalizes Q-values for unseen actions.
      * Objective:
        $$
        \mathcal{L}*{CQL} = \mathcal{L}*{Bellman} + \alpha \left( \mathbb{E}*{a \sim \pi} Q(s,a) - \mathbb{E}*{a \sim \mathcal{D}} Q(s,a) \right)
        $$
      * Encourages pessimism.
    * MOPO (Model-Based Offline Policy Optimization):

      * Penalizes model uncertainty.
      * Modified reward:
        $$
        r' = r - \lambda \cdot \text{uncertainty}
        $$

* **Models, values, and policies**

  * Model-based:

    * Learn $\hat P, \hat R$
  * Value-based:

    * Learn $Q^\pi(s,a)$:
      $$
      Q(s,a) = r + \gamma \max_{a'} Q(s',a')
      $$
  * Policy-based:

    * Directly optimize:
      $$
      \nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) A^\pi(s,a)]
      $$
  * Actor-critic combines both.

* **Computational vs data efficiency**

  * Model-free:

    * High data cost
    * Low computation per step
  * Model-based:

    * Higher computation (planning)
    * Better data efficiency
  * Tradeoff:
    $$
    \text{Total cost} = \text{environment samples} + \text{compute cost}
    $$
  * Real-world systems (robotics, plasma control) favor **data efficiency** over compute efficiency.
