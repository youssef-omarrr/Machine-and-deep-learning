import torch
from torch import nn
from torch.distributions import Categorical
import torch.optim as optim

class ActorCriticAgent(nn.Module):
    """
    Advantage Actor-Critic (A2C) agent for discrete action spaces.

    The agent uses two networks:
    1. Actor: outputs a probability distribution over actions (policy network)
    2. Critic: estimates the value of a state (value network)

    The Actor is updated using the advantage (TD error) provided by the Critic,
    while the Critic is trained to minimize the difference between its value
    prediction and the observed return (TD target).
    """
    def __init__(self, n_states, n_actions, hidden_size=128, 
                actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        super().__init__()

        self.n_actions = n_actions
        self.gamma = gamma  # Discount factor for future rewards

        # Actor network: predicts a probability distribution over actions (policy-based)
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)  # convert logits to probabilities
        )

        # Critic network: predicts the value of a given state (value-based)
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # single value for state
        )

        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        """
        Selects an action according to the policy (actor) probabilities.

        Args:
            state (np.array or list): current state of the environment

        Returns:
            int: chosen action index
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        probs = self.actor(state)  # get action probabilities
        dist = Categorical(probs)  # create a categorical distribution
        action = dist.sample()     # sample an action from the distribution

        # Store values for later learning
        self.last_log_prob = dist.log_prob(action)  # log probability of chosen action
        self.last_state_value = self.critic(state)  # value of current state (used in advantage calculations)

        return action.item()  # return as integer for env

    def update(self, reward, next_state, done):
        """
        Updates both Actor and Critic networks using the TD error (advantage).

        Args:
            reward (float): reward received after taking the last action
            next_state (np.array or list): next state after the action
            done (bool): whether the episode has ended
        """
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        # Predict the value of the next state; 0 if episode ended
        next_value = self.critic(next_state) if not done else torch.tensor([[0.0]])

        # TD target = reward + discounted value of next state
        td_target = reward + self.gamma * next_value
        
        # TD error = difference between target and predicted value (Advantage)
        delta = td_target - self.last_state_value

        # Critic loss = squared TD error (MSE)
        critic_loss = delta.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss = negative log prob of action scaled by *advantage*
        actor_loss = -self.last_log_prob * delta.detach()  # detach delta to stop backprop through critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
