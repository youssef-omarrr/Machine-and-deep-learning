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