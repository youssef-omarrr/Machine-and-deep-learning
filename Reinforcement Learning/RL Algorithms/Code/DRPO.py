import torch
from torch import nn
import torch.optim as optim

class GRPOAgent(nn.Module):
    """
    Template for GRPO (Generalized REINFORCE with Policy Optimization).

    GRPO is a policy-gradient method designed for large models (LLMs) 
    and RLHF-like setups.

    Core components:
    - Policy model
    - Reference model (frozen)
    - Reward model
    - Advantage estimator
    - GRPO loss combining REINFORCE-style gradients with PPO-style clipping
    """

    def __init__(self, policy_model, reference_model, reward_model,
                 clip_eps=0.2, gamma=1.0, lr=1e-5):
        """
        Args:
            policy_model: trainable model (e.g., transformer)
            reference_model: frozen model for KL penalty
            reward_model: model that outputs reward scalar for generated text
        """
        super().__init__()

        # Store models
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model

        # Save hyperparameters
        self.gamma = gamma
        self.clip_eps = clip_eps

        # Optimizer for the policy model only
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)

        # Storage for prompt→generated sequences and rewards
        self.reset_buffer()


    def reset_buffer(self):
        """
        Clears stored trajectories:
        - prompts
        - generated outputs
        - log_probs from policy
        - log_probs from reference model
        - reward scores
        """
        self.prompts = []
        self.outputs = []
        self.policy_logprobs = []
        self.ref_logprobs = []
        self.rewards = []


    def generate(self, prompt):
        """
        Runs the policy model to generate an output sequence.
        Must store:
        - model output
        - log probabilities of generated tokens

        Also compute the reference model log_probs for KL computation.
        """
        pass  # TODO: Generate sequence + store logprobs for both models


    def compute_rewards(self):
        """
        Uses reward model to compute reward score for each generated output.

        Store:
        - reward per sample
        """
        pass


    def compute_advantages(self):
        """
        Computes advantages for each sequence.

        GRPO advantage often includes:
        - (reward - baseline)
        - KL penalty between policy and reference model

        Keep structure empty for your implementation.
        """
        pass


    def update(self):
        """
        Performs a GRPO update step:

        1. Compute advantages
        2. Recompute policy log_probs for generated sequences
        3. Compute probability ratio:
           r = exp(logπ_new - logπ_old)
        4. Compute clipped GRPO objective (similar to PPO)
        5. Backprop and update policy model

        After finishing, clear buffer.
        """
        pass
