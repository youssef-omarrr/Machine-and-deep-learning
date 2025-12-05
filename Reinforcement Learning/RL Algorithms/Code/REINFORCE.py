import torch
from torch import nn
from torch.distributions import Categorical

class REINFORCEAgent(nn.Module):
    """
    A simple implementation of the REINFORCE (Vanilla Policy Gradient) algorithm.

    This agent:
    - Uses a policy network to output action probabilities.
    - Samples actions stochastically using a Categorical distribution.
    - Stores rewards and log-probabilities during an episode.
    - Updates the *policy* using the REINFORCE loss at the end of each episode.
    """

    def __init__(self, n_states, n_actions, hidden_size=128, lr=1e-3):
        """
        Initialize the policy network and optimizer.

        Args:
            n_states (int): Number of input features / dimensions of state space.
            n_actions (int): Number of discrete actions.
            hidden_size (int, optional): Number of neurons in hidden layers. Default=128
            lr (float, optional): Learning rate for Adam optimizer. Default=1e-3
        """
        super().__init__()

        # Policy network: maps state -> action probabilities
        self.policy_net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)  # outputs probabilities for all actions
        )

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # Memory for a single episode
        self.log_probs = []  # log probabilities of actions taken
        self.rewards = []    # rewards collected

    def take_action(self, state):
        """
        Sample an action according to the policy network.

        Args:
            state (array-like): Current environment state.

        Returns:
            int: Selected action index.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_net(state)
        dist = Categorical(probs)  # stochastic policy
        action = dist.sample()      # sample an action based on probability distribution

        # Store log probability for later policy gradient calculation
        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward):
        """
        Store the reward received after taking an action.

        Args:
            reward (float): Reward from environment.
        """
        self.rewards.append(reward)

    def finish_episode(self, gamma=0.99):
        """
        Update the policy network using the REINFORCE algorithm.

        Steps:
        1. Compute discounted returns (future rewards) G_t for each timestep.
        2. Normalize returns (optional but helps training stability).
        3. Compute loss = -sum(log_prob * G_t) across the episode.
        4. Backpropagate and update network parameters.
        5. Clear episode memory.

        Args:
            gamma (float, optional): Discount factor for future rewards. Default=0.99
        """
        returns = []
        G = 0

        # Compute discounted return backwards
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)  # insert at the front

        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        loss = 0
        for log_prob, Gt in zip(self.log_probs, returns):
            loss += -log_prob * Gt  # REINFORCE loss

        # Backpropagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Clear memory for next episode
        self.log_probs = []
        self.rewards = []

    def forward(self, state):
        """
        Forward pass through the policy network.

        Args:
            state (array-like): Current environment state.

        Returns:
            Tensor: Action probabilities.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.policy_net(state)


"""
Example use case:

agent = REINFORCEAgent(n_states=env.observation_space.shape[0], 
                        n_actions=env.action_space.n)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_reward(reward)
        state = next_state
    agent.finish_episode(gamma=0.99)
"""
