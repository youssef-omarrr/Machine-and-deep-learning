import torch
import random


class QLearning:
    """
    A minimal tabular Q-learning agent implemented using PyTorch tensors.

    This class:
    - Stores a Q-table (state × action).
    - Computes Bellman targets.
    - Computes TD error.
    - Updates Q-values using the Q-learning update rule.
    - Supports epsilon-greedy action selection.
    """

    # 1. Initialization
    # ------------------

    def __init__(self, n_states, n_actions):
        """
        Initialize the Q-table with zeros.

        Args:
            n_states (int): Total number of discrete states.
            n_actions (int): Total number of discrete actions.
        """
        self.Q = torch.zeros((n_states, n_actions))


    # 2. Core Q-learning math (Bellman -> TD -> update)
    # ---------------------------------------------------

    def calculate_reward(self, reward, gamma, next_state):
        """
        Bellman target:

            target = reward + gamma * max_a Q(next_state, a)
        """
        max_next_Q = torch.max(self.Q[next_state])
        return reward + gamma * max_next_Q


    def TD_error(self, reward, state, action, gamma, next_state):
        """
        Temporal-difference (TD) error:

            TD = target - Q(state, action)
        """
        target = self.calculate_reward(reward, gamma, next_state)
        old_Q = self.Q[state, action]
        return target - old_Q


    def update(self, reward, state, next_state, action, alpha, gamma):
        """
        Q-learning update rule:

            Q(s, a) <- Q(s, a) + alpha * TD_error
        """
        td = self.TD_error(reward, state, action, gamma, next_state)
        self.Q[state, action] += alpha * td


    # 3. Action selection (policy)
    # -------------------------------

    def select_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.

        With probability epsilon → random action.
        Otherwise → greedy action.
        """
        if random.random() < epsilon:
            return random.randrange(self.Q.shape[1])  # exploration
        
        return best_action(state)     # exploitation


    def best_action(self, state):
        """Greedy action for evaluation (no exploration)."""
        return torch.argmax(self.Q[state]).item()


    # 4. Utilities
    # -------------

    def reset_Q(self):
        """Reset the Q-table to all zeros."""
        self.Q.zero_() 


    def q_value(self, state):
        """Return V(s) = max_a Q(s,a)."""
        return torch.max(self.Q[state]).item()


    def forward(self, state):
        """
        Return all Q-values for a given state.
        Equivalent to querying the Q-table row.
        """
        return self.Q[state]


def train_q_learning(agent, env, episodes, alpha, gamma, epsilon):
    """
    Train a QLearning agent on a given environment.

    Args:
        agent (QLearning): The Q-learning agent.
        env: OpenAI Gym-like environment.
        episodes (int): Number of episodes to run.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
    """
    for ep in range(episodes):
        state = env.reset()
        done = False
        terminated = False
        truncated = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(reward, state, next_state, action, alpha, gamma)
            state = next_state
