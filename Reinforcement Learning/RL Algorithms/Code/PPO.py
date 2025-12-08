import torch
from torch import nn
from torch.distributions import Categorical
import torch.optim as optim

class PPOAgent(nn.Module):
    """
    Proximal Policy Optimization (PPO) agent with clipped objective.

    Usage pattern:
        agent = PPOAgent(n_states, n_actions, ...)
        action = agent.select_action(state)                    # each env step
        agent.buffer_rewards.append(reward); agent.buffer_dones.append(done)
        ...
        agent.update()                                         # when ready to update

    This implementation:
    - Stores trajectories in simple lists, converts to tensors at update time.
    - Computes discounted returns (Monte Carlo) and advantages = returns - values.
    - Performs K_epochs of optimization using the clipped surrogate objective.
    - Uses entropy bonus, value loss coefficient, and advantage normalization.
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
            nn.Linear(hidden_size, 1) # single value for state
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
        # Prepare state tensor on the correct device
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # -> unsqueeze for batch dimension
        
        # Forward through policy
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        
        # Save necessary items (store detached CPU/float tensors where appropriate)
        logprob = dist.log_prob(action).detach()                 # (1,)
        
        with torch.inference_mode():
            value = self.critic(state).detach()               # (1,1)

        # Convert to 1-D tensors for storage (still on device)
        self.buffer_states.append(state.squeeze(0))       # (n_states,)
        self.buffer_actions.append(action.squeeze(0))            # scalar tensor
        self.buffer_logprobs.append(logprob.squeeze(0))          # scalar tensor
        self.buffer_values.append(value.squeeze(0))              # scalar tensor (value)
        
        # rewards/dones should be appended by the environment loop externally
        # (agent.buffer_rewards.append(r); agent.buffer_dones.append(done))
        
        return int(action.item())

    def compute_returns_and_advantages(self, last_value=0.0):
        """
        Compute discounted returns and advantages (simple Monte-Carlo returns).
        Optionally supply last_value bootstrap for incomplete episodes.

        Returns:
            returns: tensor shape (T,) on device
            advantages: tensor shape (T,) on device
        """
        # load rewards, dones, and values
        rewards = self.buffer_rewards
        dones = self.buffer_dones
        values = torch.stack(self.buffer_values)
        
        # number of stored timesteps (T)
        T = len(rewards)
        
        # init returns list -> same len as timesteps
        returns = [0.0]* T
        
        # init running return (not done yet) (with bootstrap)
        R = float(last_value)
        
        # loop through timesteps in reverse to calculate returns per timestep
        for t in reversed(range(T)):
            if dones[t]:
                R = 0.0 # -> reset return if the timestep is done
                
            R = rewards[t] + self.gamma * R
            returns[t] = R # Store return for timestep t
        
        # convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # calculate advantage
        adv = returns - values
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        
        return returns, adv
    
    def evaluate_policy(self, states, actions):
        """
        For a batch of states and actions, compute:
            - new log_probs
            - state values

        Args:
            states: tensor shape (B, n_states) on device
            actions: tensor shape (B,) on device

        Returns:
            log_probs (B,), values (B,)
        """
        # Ensure correct device and shape
        probs = self.actor(states)                    # (B, n_actions)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)            # (B,)
        values = self.critic(states).squeeze(-1)      # (B,)

        return log_probs, values

    def update(self, last_value =0.0):
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
        if len(self.buffer_states) == 0:
            return # no data to update on
        
        # Convert buffers to tensors
        states = torch.stack(self.buffer_states)  # (T, state_dim)
        actions = torch.stack(self.buffer_actions).long()       # (T,)
        old_logprobs = torch.stack(self.buffer_logprobs).to(torch.float32)  # (T,)
        
        # compute return and adv
        returns, adv = self.compute_returns_and_advantages(last_value)
        returns = returns.detach()
        adv = adv.detach()
        
        for _ in range(self.K_epochs):
            
            # Evalute new policy
            new_logprobs, values = self.evaluate_policy(states, actions)
            
            # compute prob ratios
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            # Clipped surrogate objective -> actor loss
            surr1 = ratios * adv
            surr2 = torch.clip(ratios, 1-self.clip_eps, 1+self.clip_eps) * adv
            
            actor_loss = -torch.min(surr1, surr2).mean() # -> loss is the negative
            
            # Critic loss (value regression)
            value_loss = nn.functional.mse_loss(values, returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Clear collected trajectory buffer
        self.reset_buffer()