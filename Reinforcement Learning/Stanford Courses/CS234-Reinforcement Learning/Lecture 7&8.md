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

* Max entropy inverse RL algorithm → can we recover $R$?

  * Inverse RL asks: given expert demonstrations, can we recover a reward function $R$ under which the expert policy is optimal?
  * The problem is ill-posed: many rewards induce the same optimal policy.
  * Maximum Entropy IRL resolves ambiguity by choosing the distribution over trajectories that:

    * Matches expert feature expectations
    * Has maximum entropy (least additional assumptions)
  * The principle ensures the most unbiased explanation consistent with demonstrations.

* From max entropy to prob over trajectories

  * We model a distribution over trajectories $\tau$:
    $$p(\tau) \propto \exp(R(\tau))$$
  * If reward is linear in features:
    $$R(\tau) = w^\top \phi(\tau)$$
  * Then:
    $$p_w(\tau) = \frac{1}{Z(w)} \exp(w^\top \phi(\tau))$$
  * Where $Z(w)$ is the partition function:
    $$Z(w) = \sum_{\tau} \exp(w^\top \phi(\tau))$$
  * This is analogous to an energy-based model.

* Maximizing the entropy over the prob = maximizing the likelihood of observed data under max entropy

  * Maximum entropy subject to feature matching constraints yields the exponential family form above.

  * Learning reduces to maximum likelihood estimation:
    $$\max_w \sum_{\tau \in \mathcal{D}} \log p_w(\tau)$$

  * Expanding log-likelihood:
    $$\mathcal{L}(w) = \sum_{\tau \in \mathcal{D}} w^\top \phi(\tau) - \log Z(w)$$

  * Gradient:
    $$\nabla_w \mathcal{L}(w) = \mathbb{E}*{\tau \sim \mathcal{D}}[\phi(\tau)] - \mathbb{E}*{\tau \sim p_w}[\phi(\tau)]$$

  * So learning enforces:

    * Expert feature expectations = model feature expectations

  * We can estimate $R$ by maximizing probability of our observations

    * Adjust $w$ so expert trajectories become high probability under $p_w(\tau)$.

  * State densities

    * The expected feature counts depend on state visitation distribution:
      $$d^\pi(s) = \sum_{t=0}^\infty \gamma^t P(s_t = s \mid \pi)$$
    * Matching feature expectations is equivalent to matching discounted state-action visitation frequencies.

* RLHF

  * Reinforcement Learning from Human Feedback replaces manually designed rewards with learned human preferences.

  * Agent gets input from human and from environment

    * Environment provides trajectories.
    * Humans provide feedback (preferences or rankings).
    * A reward model is trained from human data.
    * RL optimizes the policy against the learned reward.

  * DAGGER/constant teaching ← human effort → demonstrations only

    * Pure demonstration methods (e.g., DAGGER) require constant expert labeling.
    * RLHF instead scales feedback by learning a reward model from limited comparisons.

* Bradley–Terry model

  * A probabilistic model for pairwise comparisons.
  * If two trajectories $\tau_i$ and $\tau_j$ have rewards $R_i, R_j$, then:
    $$P(\tau_i \succ \tau_j) = \frac{\exp(R_i)}{\exp(R_i) + \exp(R_j)}$$
  * In RLHF, $R$ is predicted by a learned reward model $R_\phi(\tau)$.
  * Training objective:
    $$\max_\phi \sum \log \frac{\exp(R_\phi(\tau_{\text{preferred}}))}{\exp(R_\phi(\tau_{\text{preferred}})) + \exp(R_\phi(\tau_{\text{other}}))}$$

* Condorcet winner / Copeland winner / Borda winner

  * These are aggregation methods for preferences.

  * Condorcet winner:

    * A candidate that beats every other candidate in pairwise comparisons.
    * May not always exist.

  * Copeland winner:

    * Score = number of pairwise wins − losses.
    * Candidate with highest score wins.

  * Borda winner:

    * Assign points based on ranking positions.
    * Candidate with highest total points wins.

  * In RLHF context, these relate to aggregating human preferences over trajectories or responses.

* High level instantiation: RLHF pipeline

  * First step: instruction tuning

    * Supervised fine-tuning on demonstration data.
    * Optimize:
      $$\max_\theta \mathbb{E}*{(x,y)}[\log \pi*\theta(y|x)]$$

  * Second + third steps max rewards

    * Step 2: Train reward model using preference data (Bradley–Terry loss).
    * Step 3: Optimize policy using RL (often PPO) to maximize learned reward:
      $$\max_\theta \mathbb{E}*{\tau \sim \pi*\theta}[R_\phi(\tau)] - \beta D_{KL}(\pi_\theta | \pi_{\text{ref}})$$
    * KL penalty ensures policy stays close to reference model, preventing reward hacking.
