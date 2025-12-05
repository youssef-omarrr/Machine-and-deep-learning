import torch
from torch import nn
from torch.distributions import Categorical
import torch.optim as optim

class PPOAgent(nn.Module):
    """
    Template for Proximal Policy Optimization (PPO) with clipped objective.

    The agent uses:
    - Actor-Critic model (shared or separate networks)
    - A buffer to store trajectories for multiple steps
    - PPO update with clipping to prevent large policy updates
    """

    def __init__(self, n_states, n_actions, hidden_size=128,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99,
                 clip_eps=0.2, K_epochs=4):
        super().__init__()

        # Save important parameters
        self.gamma = gamma          # discount factor
        self.clip_eps = clip_eps    # clipping threshold ε
        self.K_epochs = K_epochs    # number of optimization epochs
        self.n_actions = n_actions

        # Actor network (outputs action distribution)
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

        # Critic network (outputs value estimate)
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Storage for trajectories (states, actions, log_probs, rewards, etc.)
        self.reset_buffer()


    def reset_buffer(self):
        """
        Clears all stored transitions.
        PPO collects multiple steps before doing an update.
        """
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_values = []


    def select_action(self, state):
        """
        Runs the policy network to choose an action.
        Stores required information for PPO update:
        - state
        - action
        - log probability of action
        - critic value estimate
        """
        pass  # TODO: Implement sampling from π(a|s) and storing transition info


    def compute_returns_and_advantages(self):
        """
        Computes:
        - discounted returns G_t for each timestep
        - advantages using (G_t - value_t)

        Advantage estimation can be:
        - simple (Monte Carlo)
        - or GAE (Generalized Advantage Estimation)

        For template: leave structure only.
        """
        pass


    def update(self):
        """
        Performs the PPO update using collected trajectories:
        1. Compute returns and advantages
        2. For K epochs:
            a. Evaluate new log_probs and values
            b. Compute ratio: r_t = π_new / π_old
            c. Compute clipped objective
            d. Update actor
            e. Update critic (MSE loss)

        After update, clear buffer.
        """
        pass  # TODO: Implement PPO update loop


    def evaluate_policy(self, states, actions):
        """
        Used inside PPO update:
        - Computes π(a|s), log_probs, entropy, and critic values for batches.

        Returns:
            log_probs, values, entropy
        """
        pass
