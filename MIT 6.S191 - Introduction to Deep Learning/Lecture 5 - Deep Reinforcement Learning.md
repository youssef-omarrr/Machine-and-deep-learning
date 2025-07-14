# Lecture 5: Deep Reinforcement Learning 
### [Video Link](https://www.youtube.com/watch?v=to-lHJfK4pw&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=5&ab_channel=AlexanderAmini)
### [Slides Link](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L5.pdf)



## 1. **Reinforcement Learning Setup**
    
- **Agent**, **environment**, **state** s, **action** a, **reward** r, and **policy** π(a∣s).
	
- The agent interacts with the environment to maximize cumulative reward $$R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$$
## 2. **Value Functions**
    
- _State-value_: $V^\pi(s) = \mathbb{E}_\pi [R_t | s_t = s]$
	
- _Action-value_: $Q^\pi(s, a) = \mathbb{E}_\pi [R_t | s_t = s, a_t = a]$
    
## 3. **Policy & Value Iteration**
    
- Dynamic programming approach to improving π by applying Bellman optimality updates.
    
## 4. **Function Approximation with Neural Nets**
    
- Use deep networks to approximate Q(s,a) or π(a∣s) when state-action spaces are large.
    
## 5. **Deep Q-Network (DQN)**
    
- Uses replay buffer, a target network, and gradient descent to minimize TD error:  
	$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')} \left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right] $$
- Isaac: target network weights $\theta^-$ are updated slowly (e.g., every 10k steps).
    
## 6. **Policy Gradient Methods (REINFORCE)**
    
- Directly optimize policy:  
	$$\nabla_\theta J(\theta) = \mathbb{E} \left[\nabla_\theta \log \pi_\theta(a_t | s_t) R_t\right]$$
- Use **baseline** (e.g., value function) to reduce variance:  
    $$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\,(R_t - b(s_t))\right]$$
## 7. **Actor-Critic**
    
- Combines policy (actor) and value (critic); critic estimates V(s) for better policy gradient.
    

---

## Example Implementations in PyTorch

Let's illustrate a basic REINFORCE algorithm and Q-learning with neural nets:

### 🔹 REINFORCE in PyTorch

```python
import torch, torch.nn as nn, torch.optim as optim
import gym

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

def reinforce(env_name='CartPole-v1', lr=1e-2, gamma=0.99, episodes=500):
    env = gym.make(env_name)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(episodes):
        state = env.reset()
        log_probs, rewards = [], []

        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float)
            probs = policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            state, r, done, _ = env.step(action.item())
            rewards.append(r)

        # Compute discounted rewards
        G, returns = 0, []
        for r in rewards[::-1]:
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = []
        for logp, Gt in zip(log_probs, returns):
            loss.append(-logp * Gt)
        optimizer.zero_grad()
        torch.stack(loss).sum().backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, total reward {sum(rewards)}")
```

---

### 🔹 Deep Q-Network (DQN) in PyTorch

```python
import random
from collections import deque

class QNet(nn.Module):
    def __init__(self, obs_sz, act_sz):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_sz, 128),
            nn.ReLU(),
            nn.Linear(128, act_sz)
        )
    def forward(self, x):
        return self.fc(x)

def train_dqn(env_name='CartPole-v1', epochs=200, replay_size=10000, batch_size=64,
              lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=0.995,
              target_update=10):
    env = gym.make(env_name)
    obs_sz = env.observation_space.shape[0]
    act_sz = env.action_space.n

    policy_net = QNet(obs_sz, act_sz)
    target_net = QNet(obs_sz, act_sz)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    rb = deque(maxlen=replay_size)
    eps = eps_start

    for ep in range(epochs):
        state = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qs = policy_net(torch.tensor(state, dtype=torch.float))
                    action = int(qs.argmax().item())

            next_state, reward, done, _ = env.step(action)
            rb.append((state, action, reward, next_state, done))
            state = next_state

            if len(rb) < batch_size:
                continue

            batch = random.sample(rb, batch_size)
            s, a, r, ns, d = zip(*batch)
            s = torch.tensor(s, dtype=torch.float)
            a = torch.tensor(a)
            r = torch.tensor(r, dtype=torch.float)
            ns = torch.tensor(ns, dtype=torch.float)
            d = torch.tensor(d, dtype=torch.float)

            qvals = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
            q_next = target_net(ns).max(1)[0].detach()
            target = r + gamma * q_next * (~d)
            loss = nn.functional.mse_loss(qvals, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eps = max(eps_end, eps * eps_decay)
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if ep % 20 == 0:
            print(f"Episode {ep}, eps {eps:.2f}")

    print("Training complete")
```

---


