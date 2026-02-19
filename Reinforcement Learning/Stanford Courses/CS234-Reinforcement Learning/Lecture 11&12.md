# Lecture 11

* Data efficient RL / bandits

  * **Data efficiency (sample efficiency)** = how much performance improves per environment interaction.
  * **Bandits** are the simplest setting for studying data efficiency because there is *no state transition*, only repeated action selection with stochastic rewards.
  * Bandits isolate the exploration–exploitation tradeoff.

* Compute efficiency vs sample efficiency (data)

  * *Compute efficiency*: how fast algorithm runs per update.
  * *Sample efficiency*: how many samples needed to learn near-optimal policy.
  * In many real-world problems (e.g., robotics), **data is expensive**, compute is cheap → prioritize sample efficiency.

* Evaluation criteria

  * Convergence is not guaranteed → because of the *deadly triad*

    * Deadly triad = **function approximation + bootstrapping + off-policy learning** → may cause divergence.
  * How quickly we reach optimum policy → learning speed.
  * Mistakes made along the way → measured by **regret** (key metric in bandits).

* Bandits (fast learning)

  * No states, only arms (actions).
  * At each round $t$:

    * Choose arm $a_t \in {1,\dots,K}$
    * Receive reward $r_t \sim \mathcal{D}_{a_t}$
  * Goal: maximize cumulative reward over time.

* Why greedy algorithm is not always the best / most practical

  * Greedy always picks arm with highest estimated mean.
  * If early estimates are wrong, it may never explore better arms.
  * Leads to **linear regret** in worst case.

* Regret: opportunity loss for one step

  * Let $\mu^*$ = expected reward of optimal arm.
  * If we pick arm $a_t$ with mean $\mu_{a_t}$, instantaneous regret:
    $$r_t^{\text{regret}} = \mu^* - \mu_{a_t}$$

* Total regret

  * Over $T$ steps:
    $$R_T = \sum_{t=1}^{T} (\mu^* - \mu_{a_t})$$
  * Equivalent form using counts:
    $$R_T = \sum_{a=1}^{K} \Delta_a \mathbb{E}[N_a(T)]$$
  * Where:

    * $\Delta_a = \mu^* - \mu_a$ (gap)
    * $N_a(T)$ = number of times arm $a$ selected.

* Evaluating regret:

  * Count → how many times action is taken

    * $N_a(T)$ determines regret contribution.
  * Gap → like advantage

    * $\Delta_a$ measures suboptimality.
  * Regret → function of count and gap

    * Large gap × many pulls = large regret.

* Good algorithm

  * Low counts for larger gaps

    * Should quickly eliminate bad arms.
  * Gaps are unknown → optimal action is unknown

    * Need exploration to estimate gaps.

* Regret for greedy methods can be linear

  * If greedy locks onto wrong arm:
    $$R_T = O(T)$$
  * Linear regret is bad → average regret does not vanish.
  * Advantage of linear regret (rare case): simple, low compute, but not statistically optimal.

* Types of regret bounds

  * Problem dependent

    * Depends on gaps $\Delta_a$.
    * Example (UCB):
      $$R_T = O\left(\sum_a \frac{\log T}{\Delta_a}\right)$$
  * Problem independent

    * Worst case over all problems:
      $$R_T = O(\sqrt{KT \log T})$$

* Lower bound

  * Any algorithm must suffer at least:
    $$R_T = \Omega\left(\sum_a \frac{\log T}{\Delta_a}\right)$$
  * Logarithmic regret is optimal up to constants.

* Optimism in face of uncertainty

  * Principle:

    * Choose action with highest *plausible upper reward*.
  * Encourages exploration automatically.
  * Key idea behind UCB.

* Upper Confidence Bounds (UCB) (fast learning)

  * Maintain estimate $\hat{\mu}_a$ and confidence interval.
  * Select arm maximizing:
    $$\hat{\mu}_a + \text{confidence bonus}$$

* UCB1 algorithm

  * Confidence bonus:
    $$\sqrt{\frac{2 \log T}{N_a(t)}}$$

  * Pseudocode:

    ```cpp
    for each arm a:
        pull once
    for t = K+1 to T:
        select arm a maximizing:
            μ̂_a + sqrt( (2 log t) / N_a )
        observe reward
        update μ̂_a and N_a
    ```

  * Explanation:

    * If arm pulled few times → large bonus → exploration.
    * If arm pulled many times → small bonus → exploitation.

* Confidence level (as a metric)

  * Derived using Hoeffding inequality:
    $$P(|\hat{\mu}_a - \mu_a| > \epsilon) \le 2 e^{-2 N_a \epsilon^2}$$
  * Confidence term shrinks with more samples.

* Regret bounds for UCB multi-armed bandit sketch

  * If confidence intervals hold, suboptimal arm chosen only if:

    * Its upper bound overlaps optimal arm.
  * Happens when:
    $$\sqrt{\frac{\log T}{N_a}} \ge \Delta_a$$
  * So:
    $$N_a = O\left(\frac{\log T}{\Delta_a^2}\right)$$
  * Plug into regret → logarithmic regret.

* If your confidence bounds hold and you use them to take decisions, then the only time the decision is wrong is when the confidence bounds are large enough that it overwhelms the gap

  * Mistake condition:
    $$\text{bonus} \ge \Delta_a$$
  * Once enough samples collected, this no longer holds → arm eliminated.

* An alternative is to always select the arm with the highest lower bound (why?)

  * This is *conservative optimism*.
  * Lower bound ensures guaranteed performance.
  * However, selecting highest upper bound ensures faster learning of optimal arm.

# Lecture 12

* K-armed multi-armed bandits = single state MDP with K actions

  * Equivalent to MDP with:

    * One state
    * K actions
    * No transitions

* Bayesian bandits (fast learning)

  * Maintain posterior over reward parameters.
  * Use prior knowledge to guide exploration.

* Probably Approximately Correct (PAC) algorithm → to initialize weight

  * PAC guarantees:

    * With probability $1-\delta$, find $\epsilon$-optimal arm after finite samples.
  * Sample complexity bound:
    $$O\left(\frac{K}{\epsilon^2}\log\frac{1}{\delta}\right)$$

* Most PAC algorithms based on optimism or Thompson sampling

  * Optimism → UCB
  * Thompson sampling → probability matching.

* Greedy bandit algorithms vs optimistic initialization

  * Greedy fails without exploration.
  * Optimistic initialization sets high initial $\hat{\mu}_a$ → forces exploration early.

* Refresh on Bayesian inference

  * Prior: $P(\theta)$
  * Likelihood: $P(D|\theta)$
  * Posterior:
    $$P(\theta|D) \propto P(D|\theta) P(\theta)$$

* Bayesian bandits → exploit prior knowledge of reward function

  * Example: Bernoulli reward with Beta prior:
    $$\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$$
  * Posterior updated after observing reward.

* Probability matching → select action based on probability of that action being optimal

  * Instead of selecting highest mean, sample from posterior and act greedily w.r.t sample.
  * Leads to Thompson sampling.

* Thompson sampling algorithm

  * Pseudocode (Bernoulli bandit):

	```cpp
	for each arm a:
		initialize α_a, β_a
	for t = 1 to T:
		for each arm a:
			sample θ_a ~ Beta(α_a, β_a)
		select arm a_t = argmax θ_a
		observe reward r
		update:
			α_a += r
			β_a += (1 - r)
	```

  * Explanation:

    * Randomized exploration proportional to posterior uncertainty.
    * Naturally balances exploration and exploitation.

* Regret and Bayesian regret

  * Regret: expectation over algorithm randomness.
  * Bayesian regret: expectation over prior distribution of problem parameters.
  * Thompson sampling achieves:
    $$R_T = O(\sqrt{KT \log T})$$

* Gittins index for Bayesian bandits (index policy)

  * Assign each arm an index depending only on its posterior.
  * Select arm with highest index.
  * Optimal for discounted infinite-horizon Bayesian bandits.
  * Computationally complex but theoretically optimal.
