import torch
from torch import nn
import random
import copy
from collections import deque


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) agnet.

    This class implements:
    - Q-network and target network
    - Epsilon-greedy action selection
    - Experience replay buffer
    - Target network update
    - Forward pass for Q-values
    """

    # 1. Initialization
    # -------------------
    def __init__(self, n_states, hidden_size, n_actions,
                buffer_size=10000, lr=1e-3):
        """
        Initialize DQN agent.

        Args:
            n_states (int): Number of input features / state dimensions
            hidden_size (int): Number of neurons in hidden layers
            n_actions (int): Number of discrete actions
            buffer_size (int, optional): Maximum replay buffer size
            lr (float, optional): Learning rate for Adam optimizer
        """
        super().__init__()

        # Q-network (approximates Q-values)
        self.Q_network = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        # Target network (used for stable updates)
        self.target_network = copy.deepcopy(self.Q_network)
        self.target_network.eval()  # freeze target network

        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Optimizer and loss function
        self.optim = torch.optim.Adam(self.Q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # 2. Action selection
    # ---------------------
    def select_action(self, state, epsilon):
        """
        Select action using epsilon-greedy policy.

        Args:
            state (array-like): Current environment state
            epsilon (float): Probability of choosing a random action

        Returns:
            int: Selected action index
        """
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randrange(self.Q_network[-1].out_features)
        else:
            # Exploitation: choose the best action
            return self.best_action(state)

    def best_action(self, state):
        """
        Greedy action selection (no exploration).

        Args:
            state (array-like): Current environment state

        Returns:
            int: Action with highest Q-value
        """
        with torch.inference_mode():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.Q_network(state_tensor)
            return torch.argmax(q_values).item()

    # 3. Replay buffer
    # -----------------
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state, action, reward, next_state, done: Experience tuple
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        """
        Sample a random batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample

        Returns:
            list: Sampled batch of transitions
        """
        return random.sample(self.replay_buffer, batch_size)
    
    # 4. Losses and Backpropagation 
    # -------------------------------
    def update_q_network(self, q_values, target_q_values):
        """
        Perform a single optimization step on the Q-network.

        This method:
        - Computes the mean squared error (MSE) loss between the predicted Q-values
        and the target Q-values.
        - Performs backpropagation and updates the Q-network parameters using the optimizer.

        Args:
            q_values (Tensor): Predicted Q-values for the actions taken (batch_size × 1)
            target_q_values (Tensor): Target Q-values computed from the Bellman equation
        """
        # Compute MSE loss
        loss = self.loss_fn(q_values, target_q_values)

        # Backpropagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    # 5. Target network update
    # --------------------------
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.Q_network.state_dict())
        
    # 6. Train on batch
    # ------------------
    def train_on_batch(self, batch_size, gamma):
        """
        Sample a batch from the replay buffer and perform one training step.

        Args:
            batch_size (int): Number of transitions to sample from the replay buffer.
            gamma (float): Discount factor for future rewards.
        """
        # Skip training if not enough samples
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch of transitions
        batch = self.sample_batch(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute current Q-values for the actions taken
        q_values = self.Q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.inference_mode():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        # update_q_network
        self.update_q_network(q_values, target_q_values)

    # 7. Forward pass
    # ----------------
    def forward(self, state):
        """
        Return Q-values for a given state or batch of states

        Args:
            state (array-like): Current environment state

        Returns:
            Tensor: Q-values for all actions
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        return self.Q_network(state_tensor)
    
    
"""
Use case:

# Store a transition in the replay buffer
agent.store_transition(state, action, reward, next_state, done)

# Train DQN on a batch
agent.train_on_batch(batch_size=64, gamma=0.99)

# Update target network periodically (e.g., every 100 steps)
if step % 100 == 0:
    agent.update_target_network()
"""
