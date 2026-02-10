# Lecture 5
* Directly parametrize the policy
	* Instead of learning a value function first and deriving a policy from it, we represent the policy explicitly as $\pi_\theta(a \mid s)$, where $\theta$ are trainable parameters (e.g., neural network weights).
	* This allows direct optimization of behavior without requiring an intermediate argmax over actions.

* Goal: find policy with highest value fn
	* The objective is to find parameters $\theta$ that maximize the expected return (value) of the policy.
	* Formally, we want to maximize the expected discounted sum of rewards:
    $$J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}\left[\sum_{t=0}^\infty \gamma^t r_t\right]$$

* Value based vs policy based vs actor critic
	* Value-based methods learn $V(s)$ or $Q(s,a)$ and derive a policy indirectly (e.g., Q-learning).
	* Policy-based methods directly optimize $\pi_\theta(a \mid s)$ without explicitly learning a value function.
	* Actor–critic methods combine both: the actor is the policy $\pi_\theta$, and the critic estimates $V^\pi(s)$ or $Q^\pi(s,a)$ to guide learning.

* Example: human in the loop exoskeleton optimization
	* In such systems, the reward may come from human feedback and may be non-differentiable or delayed.
	* Policy gradient methods are suitable because they do not require differentiating the reward function or the environment dynamics.

* Policy gradient → search for local max in value function
	* Policy gradient methods perform gradient ascent on $J(\theta)$.
	* Since $J(\theta)$ is generally non-convex, the algorithm converges to a local maximum rather than a global one.

* Initial value function is expected reward → can be re-expressed as → $v = \sum \pi \times Q$
	* The state value function under policy $\pi$ is:
    $$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}[Q^\pi(s,a)]$$
	* This expresses the value as a policy-weighted average of action-values.

* Goal is to find the policy parameter $\theta$ → $\arg\max v = \arg\max \sum P R$
	* The optimization objective can be written as:
	$$\theta^* = \arg\max_\theta \mathbb{E}*{\tau \sim \pi*\theta}[R(\tau)]$$
	* Here, $R(\tau)$ is the total return of a trajectory $\tau$.

* Take gradient wrt $\theta$ → likelihood ratio
	* The gradient of the objective is:
	$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}*{\tau \sim \pi*\theta}[R(\tau)]$$
	* Using the likelihood ratio trick:
    $$\nabla_\theta J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}\left[\nabla_\theta \log p_\theta(\tau) , R(\tau)\right]$$

* Decomposing the trajectories into states and actions
	* The trajectory probability factorizes as:
	$$p_\theta(\tau) = \rho(s_0)\prod_{t=0}^\infty \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$$
	* Since environment dynamics do not depend on $\theta$, only the policy terms contribute to the gradient.

* Score function → prob of action is proportional to expo weight
	* The score function is $\nabla_\theta \log \pi_\theta(a \mid s)$.
	* For a softmax policy:
$$\pi_\theta(a \mid s) = \frac{e^{f_\theta(s,a)}}{\sum_{a'} e^{f_\theta(s,a')}}$$
	* Actions with higher scores $f_\theta(s,a)$ get exponentially higher probability.

* Softmax policy vs Gaussian policy
	* Softmax policies are typically used for discrete action spaces.
	* Gaussian policies are used for continuous actions:
    $$\pi_\theta(a \mid s) = \mathcal{N}(a \mid \mu_\theta(s), \Sigma_\theta(s))$$

* Likelihood ratio / score function → policy fn must be differentiable but reward fn doesn’t have to be, doesn’t have to be Markov
	* The gradient only requires $\nabla_\theta \log \pi_\theta(a \mid s)$.
	* The reward can be discontinuous or unknown, and the state does not need to satisfy the Markov property.

* Score function gradient estimator: intuition
	* Trajectories with higher return are reinforced by increasing the log-probability of the actions that produced them.
	* Poor trajectories decrease the probability of their actions.

* ![](PastedImage-74.png)
	* The image illustrates how gradients push probability mass toward better-performing trajectories based on observed returns.

* Unbiased, but very noisy (high variance) → fixes: temporal structure & baselines
	* The estimator is unbiased:
	$$\mathbb{E}[\nabla_\theta \hat{J}] = \nabla_\theta J$$
	* However, variance is high due to long trajectories and stochastic rewards.
	* Variance reduction techniques include using per-timestep rewards and subtracting a baseline.

* REINFORCE algorithm → likelihood ratio and temporal structure
	* Uses per-step returns:
	$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) , G_t\right]$$
	* Where $G_t = \sum_{k=t}^\infty \gamma^{k-t} r_k$.

* REINFORCE pseudocode
	* Algorithm:
	```cpp
	initialize θ
	for each episode:
		sample trajectory τ using π_θ
		for each timestep t in τ:
			compute return G_t
			θ ← θ + α ∇_θ log π_θ(a_t | s_t) G_t
	```
	* Explanation: actions are reinforced proportionally to their future return.

* Baseline goal → converge as quickly as possible to local optima
	* A baseline reduces variance without changing the expected gradient, leading to faster and more stable convergence.

# Lecture 6

* PPO
	* Proximal Policy Optimization is a policy gradient method designed to improve stability and sample efficiency by limiting policy updates.

* Baseline $b(s)$: only a function of state (not $\theta$ or $a$)
	* Commonly chosen as an estimate of $V^\pi(s)$.
	* This ensures unbiasedness of the gradient.

* For any choice of $b$ → gradient estimator is unbiased
	* Because:
    $$\mathbb{E}*{a \sim \pi*\theta}[\nabla_\theta \log \pi_\theta(a \mid s) b(s)] = 0$$

* Near optimum is expected return
	* At convergence, the baseline often approximates the expected return, minimizing variance.

* “Vanilla” policy gradient algorithm
	* Refers to REINFORCE without trust regions, clipping, or advanced variance reduction.

* Actor → policy / critic → $v/q$
	* The actor updates $\pi_\theta(a \mid s)$.
	* The critic learns $V^\pi(s)$ or $Q^\pi(s,a)$ to evaluate actions.

* Critic can select any blend between TD and MC estimators
	* Monte Carlo: unbiased, high variance.
	* Temporal Difference (TD): biased, low variance.
	* A weighted combination trades off bias and variance.

* Advantage estimators → subtract baseline from blended estimator → $A_1$: low variance high bias, $A_\infty$: high variance low bias
	* Advantage:
	$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$
	* Generalized Advantage Estimation (GAE) interpolates between TD and MC.

* Problems with policy gradient methods
	* Sample efficiency is poor due to on-policy data usage.
	* Distance in parameter space $\neq$ distance in policy behavior.

* Use old data to take multiple gradient steps
	* Off-policy reuse improves efficiency but risks destabilizing learning.

* Policy gradient algorithms are stochastic gradient ascent
	* Updates are noisy estimates of $\nabla_\theta J(\theta)$ using sampled trajectories.

* Policy performance bounds
	* Theoretical bounds relate performance improvement to policy divergence.

* Relative policy performance (in terms of discounted future rewards)
	* Improvement depends on expected advantages under the old policy.

* If policies are close in KL-divergence → the approximation is good
	* Trust-region methods ensure updates stay within a small KL neighborhood.

* KL-divergence → pdf $P$ and $Q$ over discrete random variable
	* Defined as:
    $$D_{KL}(P | Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

* KL-divergence between policies
	* For policies $\pi_{\text{old}}$ and $\pi_{\text{new}}$:
    $$D_{KL}(\pi_{\text{old}} | \pi_{\text{new}}) = \mathbb{E}*{s,a \sim \pi*{\text{old}}}\left[\log \frac{\pi_{\text{old}}(a \mid s)}{\pi_{\text{new}}(a \mid s)}\right]$$

* Adaptive KL penalt
	* Adds a penalty term $\beta D_{KL}$ to the objective and adjusts $\beta$ to control update size.

* PPO algorithm with adaptive KL penalty
	* Objective:
    $$L(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A - \beta D_{KL}\right]$$

* Clipped objective
	* PPO uses:
	$$L^{\text{clip}}(\theta) = \mathbb{E}\left[\min(r(\theta)A,\ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A)\right]$$
	* This prevents excessively large policy updates and improves training stability.
