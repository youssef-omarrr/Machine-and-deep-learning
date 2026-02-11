# Lecture 7

* Telescoping sum in n-step advantage estimators

  * n-step advantage estimators are built from TD errors:
    $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
  * The k-step return is:
    $$G_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k})$$
  * The k-step advantage is:
    $$A_t^{(k)} = G_t^{(k)} - V(s_t)$$
  * Expanding this using TD errors produces a telescoping sum:
    $$A_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$$
  * The telescoping property arises because intermediate $V(s)$ terms cancel, leaving only the first and last value terms.

* GAE, Generalized Advantage Estimator → exponentially weighted average of k-step estimators

  * GAE defines advantage as:
    $$A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
  * This is equivalent to an exponentially weighted average of k-step advantages:
    $$A_t^{\text{GAE}(\lambda)} = (1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} A_t^{(k)}$$
  * $\lambda \in [0,1]$ controls the bias-variance tradeoff:

    * $\lambda = 0$: 1-step TD (low variance, high bias)
    * $\lambda \to 1$: Monte Carlo (low bias, high variance)

* PPO uses a truncated version of GAE

  * In practice, trajectories are finite, so the infinite sum is truncated at episode end or rollout horizon.
  * Advantage is computed backward through collected rollout data using:
    $$A_t = \delta_t + \gamma \lambda A_{t+1}$$

* Monotonic improvement theory → improvement from $\pi_k$ to $\pi_{k+1}$ is guaranteed

  * The performance difference lemma:
    $$J(\pi') - J(\pi) = \frac{1}{1-\gamma} \mathbb{E}*{s \sim d*{\pi'}} \mathbb{E}_{a \sim \pi'}[A^\pi(s,a)]$$
  * Trust Region Policy Optimization (TRPO) derives a lower bound:
    $$J(\pi') \ge L_\pi(\pi') - C \cdot D_{KL}^{\max}(\pi | \pi')$$
  * If the KL-divergence is constrained to be small, performance improvement is guaranteed.

* PPO improves data efficiency, uses clipping to help increase likelihood of monotonic improvement, converges to local optima

  * PPO objective:
    $$L^{\text{clip}}(\theta) = \mathbb{E}\left[\min(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$
  * Where:
    $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$
  * Clipping prevents large updates, approximating trust-region behavior.
  * Like all gradient-based methods, PPO converges to a local optimum.

* Imitation learning → very good decision policies and we would like to automate them

  * Instead of defining rewards manually, we learn policies from expert demonstrations.

* Reward shaping → manual design: brittle, implicitly specify them through demonstration

  * Hand-crafted rewards may fail to capture complex behaviors.
  * Demonstrations implicitly encode desired behavior without explicit reward engineering.

* Demonstration trajectories (state actions only) → no reward function

  * Dataset:
    $$\mathcal{D} = {(s_1,a_1), (s_2,a_2), \dots}$$
  * No reward labels are provided.

* Behaviour cloning → standard supervised ML → example: BCRNN

  * Treat imitation as supervised learning:
    $$\min_\theta \mathbb{E}*{(s,a) \sim \mathcal{D}}[-\log \pi*\theta(a|s)]$$
  * Equivalent to maximum likelihood estimation.
  * BCRNN uses a recurrent neural network to handle partial observability.

* DAGGER → challenges of BC

  * Supervised learning ignores temporal structure:

    * Small mistakes lead to states not seen in training data (distribution shift).
    * Errors compound over time.

* DAGGER dataset aggregation algorithm → get more labels from expert action along the path taken by the policy computed by behaviour cloning

  * Key idea: collect data under the learner’s policy but label with expert actions.

  * Pseudocode:

    ```cpp
    initialize D with expert demonstrations
    initialize policy π_1 using BC on D
    for i = 1 to N:
        run π_i to collect states
        query expert for correct action at visited states
        aggregate new (s, a_expert) into D
        retrain policy π_{i+1} on aggregated dataset D
    return best π_i
    ```

  * Explanation:

    * The learner visits states it is likely to encounter.
    * The expert corrects mistakes along those states.
    * This reduces distribution shift.

* Limitation → super expensive (constant supervision)

  * Requires querying expert repeatedly during training.
  * Not feasible when expert is human or costly.

* Reward learning → given state action transition, no reward function → goal: infer the reward function

  * Instead of learning policy directly, learn reward $R(s,a)$ explaining expert behavior.
  * Then solve RL with learned reward.

* Assume expert policy is optimal → there are many $R$ that make the expert’s policy optimal

  * Inverse RL is ill-posed: multiple reward functions can induce the same optimal policy.
  * Additional constraints (e.g., max-margin, entropy regularization) are required.

* Linear value function approx → linear feature reward inverse RL

  * Assume reward is linear in features:
    $$R(s) = w^\top \phi(s)$$
  * Value function:
    $$V^\pi(s) = w^\top \mathbb{E}_\pi\left[\sum_t \gamma^t \phi(s_t)\right]$$
  * Define feature expectations:
    $$\mu^\pi = \mathbb{E}_\pi\left[\sum_t \gamma^t \phi(s_t)\right]$$

* Goal: identify the weight vector given a set of demonstrations

  * Given expert feature expectations $\mu^E$, find $w$ such that:
    $$w^\top \mu^E \ge w^\top \mu^\pi \quad \forall \pi$$
  * Often solved via convex optimization (e.g., max-margin formulation).
  * The learned $w$ defines a reward under which expert behavior is optimal.

# Lecture 8
- 