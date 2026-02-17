# Lecture 9

* Examples on RLHF style algorithms

  * **RLHF-style algorithms** optimize a policy using *human preferences instead of hand-designed rewards*.
  * Core pipeline idea:

    * Collect preference data (e.g., humans choose between two outputs).
    * Train a **reward model** from preferences.
    * Optimize the policy to maximize this learned reward while staying close to a reference model.
  * This prevents direct optimization of raw human signals and allows scalable learning.

* DPO: direct preference optimization

  * **DPO (Direct Preference Optimization)** removes the explicit reward-model + RL step.
  * Instead of:

    * Train reward model → run PPO on reward
  * DPO directly optimizes the policy using preference pairs.
  * Key insight: preference learning objective can be rewritten directly as a policy objective.

* ChatGPT pipeline:

  * Unsupervised pretraining → supervised finetuning → policy finetuning

    * **Unsupervised pretraining**:

      * Train language model on next-token prediction:
        $$\max_\theta \mathbb{E}[\log \pi_\theta(x_t | x_{<t})]$$
      * Learns general language structure.
    * **Supervised fine-tuning (SFT)**:

      * Train on high-quality instruction-response pairs.
      * Objective:
        $$\max_\theta \mathbb{E}*{(x,y)}[\log \pi*\theta(y|x)]$$
    * **Policy fine-tuning (RLHF stage)**:

      * Optimize responses using human preferences.

* Bradley–Terry model → connects rewards to preference

  * Given two outputs $y_1, y_2$ with rewards $R(y_1), R(y_2)$:
    $$P(y_1 \succ y_2) = \frac{\exp(R(y_1))}{\exp(R(y_1)) + \exp(R(y_2))}$$
  * This converts reward differences into comparison probabilities.
  * Used to train reward models from pairwise preferences.

* Train reward model to minimize −ve log likelihood (binary classification problem)

  * Given preferred $y^+$ and rejected $y^-$:
    $$\mathcal{L} = - \log \frac{\exp(R_\phi(y^+))}{\exp(R_\phi(y^+)) + \exp(R_\phi(y^-))}$$
  * This is equivalent to logistic regression.
  * The reward model learns to assign higher scores to preferred outputs.

* Learn new policy ($\pi_\theta$) to achieve high reward while staying close to original model ($\pi_{\text{ref}}$)

  * Objective:
    $$\max_\theta \mathbb{E}*{y \sim \pi*\theta}[R_\phi(y)] - \beta D_{KL}(\pi_\theta | \pi_{\text{ref}})$$
  * First term → maximize reward
  * Second term → penalize deviation from pretrained model
  * $\beta$ controls trade-off.

* Closed form optimal policy

  * If reward model is fixed, the optimal solution satisfies:
    $$\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta} R(y)\right)$$
  * This shows policy is **reweighted reference model** by exponentiated reward.
  * This connection is what DPO exploits.

* Loss function on reward function + a transform between reward functions and policies = loss function on policy

  * Since optimal policy depends exponentially on reward,
  * We can substitute reward difference with log-prob differences between policy and reference.
  * DPO objective becomes:
    $$\mathcal{L}*{\text{DPO}} = - \log \sigma\left(\beta \left[ \log \frac{\pi*\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)} - \log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)} \right]\right)$$
  * No explicit reward model needed.

* How efficiently DPO trades off reward and KL

  * DPO implicitly enforces KL regularization through the log-ratio terms.
  * It directly optimizes the final objective instead of alternating reward fitting + PPO.
  * This makes it more stable and computationally cheaper.

* Why do we use KL as a metric/benchmark

  * **KL-divergence measures distribution shift.**
  * For language models:

    * Large deviation from pretrained model leads to instability and reward hacking.
  * KL regularization ensures:

    * Preserve fluency
    * Avoid over-optimization
    * Maintain alignment with pretrained knowledge

* DPO vs PPO (especially in reward hacking)

  * PPO:

    * Optimizes learned reward via policy gradient.
    * May exploit reward model weaknesses (reward hacking).
  * DPO:

    * Optimizes preference data directly.
    * Less indirect optimization → often more stable.
  * DPO avoids training a separate reward model that can be gamed.

* PPO is weaker optimizer (why?)

  * PPO is **on-policy** → discards old data.
  * Uses clipped objective → restricts update size.
  * Gradient is noisy and high variance.
  * DPO is closer to supervised learning → more stable gradients.

* Weight averaging in pretrained optimizer improves robustness

  * Techniques like **EMA (Exponential Moving Average)** or SWA smooth parameter updates.
  * Reduces sharp minima and overfitting.
  * Improves generalization and stability.

# Lecture 10

* Learning from old data / offline RL

  * **Offline RL**: learn policy using fixed dataset $\mathcal{D}$ without interacting with environment.
  * No exploration allowed.
  * Challenge: avoid selecting actions not supported by data.

* Counterfactual / batch RL

  * Also called **batch RL**.
  * We must estimate value of actions never actually taken in dataset → high extrapolation error risk.

* Better dynamic/reward models may not lead to better policies for future use → due to model misspecification

  * Small modeling errors compound during planning.
  * Learned model may be accurate on data distribution but inaccurate outside it.
  * Planning amplifies these errors.

* Fitted Q Evaluation (FQE)

  * Used to estimate value of a policy from offline data.

  * Iteratively fit Q-function using Bellman backup:
    $$Q_{k+1}(s,a) = r + \gamma \mathbb{E}_{a' \sim \pi}[Q_k(s',a')]$$

  * No policy improvement, only evaluation.

  * Pseudocode:

	``` cpp
	initialize Q
	repeat:
		minimize over Q:
			(Q(s,a) - (r + γ E_{a'~π} Q(s',a')))²
	return Q
	```

* FQE vs DQN

  * **FQE**: evaluates fixed policy (no max operator).
  * **DQN**: learns optimal policy using:
    $$Q(s,a) = r + \gamma \max_{a'} Q(s',a')$$
  * DQN is control; FQE is evaluation.

* Importance sampling → unbiased estimator of true expectation under desired distribution

  * If data from behavior policy $\mu$, but we want expectation under $\pi$:
    $$\mathbb{E}*{\pi}[f] = \mathbb{E}*{\mu}\left[\frac{\pi(a|s)}{\mu(a|s)} f\right]$$
  * Ratio:
    $$w = \frac{\pi(a|s)}{\mu(a|s)}$$

* Per-decision importance sampling (PDIS)

  * Apply importance weights per timestep:
    $$\hat{V} = \sum_t \gamma^t \left( \prod_{i=0}^t \frac{\pi(a_i|s_i)}{\mu(a_i|s_i)} \right) r_t$$
  * Reduces variance compared to full-trajectory weighting.

* Importance sampling variance

  * Variance grows exponentially with horizon.
  * If policies differ significantly, weights explode.
  * Main limitation of offline evaluation.

* Pessimistic offline learning

  * Idea: avoid overestimating unseen actions.
  * Add penalty for actions outside dataset distribution.
  * Conservative Q-Learning (CQL) example:
    $$\min_Q \alpha \left(\mathbb{E}*{a \sim \pi}[Q(s,a)] - \mathbb{E}*{a \sim \mathcal{D}}[Q(s,a)]\right)$$
  * Push down Q-values of unseen actions.

* Use pessimistic value for insufficient data

  * When uncertain, underestimate value.
  * Prevents selecting risky, unsupported actions.
  * Leads to safer, more stable offline policies.
