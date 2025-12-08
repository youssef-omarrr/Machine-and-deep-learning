import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer
# --------------
# Simple FIFO buffer storing transitions for off-policy learning.
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # store a transition tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # sample a random minibatch and convert to tensors on the correct device
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.tensor(state, dtype=torch.float32, device=device),
                torch.tensor(action, dtype=torch.float32, device=device),
                torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1),
                torch.tensor(next_state, dtype=torch.float32, device=device),
                torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


# Critic Network (Q-function)
# -----------------------------
# Learns Q(s,a). Two critics are used in SAC to reduce overestimation bias.
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # concatenate state and action as input to Q-network
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Actor Network (Gaussian Policy)
# ---------------------------------
# Outputs mean and log_std for a Gaussian; samples via reparameterization.
class GaussianPolicy(nn.Module):
    """
    Actor network for SAC (Soft Actor-Critic) that outputs a Gaussian policy.

    Differences between SAC and DDPG:

    1. SAC:
       - Uses a **stochastic policy**: outputs mean and standard deviation of a Gaussian.
                -> gives a probability distribution over actions, so the action can vary each time you sample
       - Samples actions via the **reparameterization trick** for differentiability.
       - Includes **entropy regularization** to encourage exploration.
       - Handles uncertainty better in continuous action spaces.
    
    2. DDPG:
       - Uses a **deterministic policy**: directly outputs an action. 
                -> Every time you give the same state 's' to the policy, it produces the same action 'a'
       - No sampling is involved; action = μ(s).
       - Relies on exploration noise added externally (e.g., Ornstein-Uhlenbeck or Gaussian noise).
       - Can be more sample-efficient but less robust to function approximation errors.

    In short: SAC = stochastic + entropy, DDPG = deterministic + external noise.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        # clamp log_std to avoid numerical issues
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        # returns mean and clipped log-std
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state): # -> returns action and its log probabilty
        # sample action using reparameterization trick for backprop through stochasticity
        mean, log_std = self(state)
        std = log_std.exp() # -> converts log_std to std
        
        # create gaussian distribution with mean and std
        normal = torch.distributions.Normal(mean, std)
        
        # reparameterization trick
        """
        Normally, sampling from a distribution is not differentiable.
        rsample(): allows us to sample actions while keeping gradients, so we can backprop through the stochastic action.
        """
        x_t = normal.rsample()  
        
        # squash to action range -> [-1, 1]
        action = torch.tanh(x_t) * self.max_action  
        
        # correction for tanh squashing in log-prob (see SAC paper)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


# SAC Agent
# -------------
# Implements core SAC algorithm: actor, two critics, target networks, and updates.
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update rate for target networks
        self.alpha = alpha  # entropy temperature (can be learned; here fixed)

        # Actor
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critics (two for clipped double Q-learning)
        self.critic1 = QNetwork(state_dim, action_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim).to(device)
        # target networks initialized from critics
        self.critic1_target = QNetwork(state_dim, action_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        # choose action. In evaluate mode use mean (deterministic), otherwise sample.
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.actor.max_action
            return action.cpu().detach().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        # sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # compute target Q value using target networks and policy entropy term
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            
            # take the minimum to reduce overestimation bias
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # Critic update: MSE to target
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update: maximize expected Q + entropy (here minimized as loss)
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        # actor loss = alpha * log_pi - Q (we minimize this)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets (polyak averaging)
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



# Training Loop (Example)
# -----------------------------
# Note: this is a simple example loop; in practice handle Gym API differences (obs,info) and action shapes.
def train_sac(env, agent, replay_buffer, episodes=1000, batch_size=256, start_steps=10000):
    total_steps = 0
    for ep in range(episodes):
        # depending on gym version, env.reset() may return (obs, info)
        state, done, ep_reward = env.reset(), False, 0
        while not done:
            # initial random exploration steps before using the policy
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # step environment and store transition
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

            # update after enough samples collected
            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

        # print episode summary (monitor training)
        print(f"Episode {ep} Reward: {ep_reward:.2f}")

