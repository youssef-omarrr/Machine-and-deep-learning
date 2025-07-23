# Lecture 5: Deep Reinforcement Learning 
### [Video Link](https://www.youtube.com/watch?v=to-lHJfK4pw&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=5&ab_channel=AlexanderAmini)
### [Slides Link](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L5.pdf)


## 1. **Reinforcement Learning Setup**

![Alt text](imgs/Pasted%20image%2020250715235813.png)
	
- **Agent**: The agent is the decision-maker that interacts with the environment to achieve a goal.
    
- **Environment**: Everything outside the agent that responds to the agent’s actions and gives feedback.
    
- **State (s)**: A representation of the current situation or configuration of the environment.
    
- **Action (a)**: A move or decision the agent makes based on the current state.
    
- **Reward (r)**: A numerical feedback signal the agent gets from the environment after taking an action.
    
- **Policy (π(a∣s))**: A strategy or rule that defines the probability of taking action **a** when in state **s**.
	
- The agent interacts with the environment to maximize cumulative reward $$R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$$
![RL Flow](imgs/Pasted%20image%2020250716000304.png)
## 2. **Learning Algorithms**
![Learning Algorithms](imgs/Pasted%20image%2020250716000949.png)
    
## 3. **Value learning**
- _Q-Function_: represents the *expected* total *future* reward the agent will get by taking action **a** in state **s** and then following policy **π** afterward. It helps the agent evaluate how good a specific action is in a given state. $$Q^\pi(s, a) = \mathbb{E}_\pi [R_t | s_t = s, a_t = a]$$
- The expression $a = \arg\max Q(s, a)$ means the agent chooses the action **a** that gives the highest Q-value in state **s**. In other words, the agent picks the best possible action based on its current knowledge.    
## 4. **Deep Q-Network (DQN)**

![Deep Q-Network Diagram](imgs/Pasted%20image%2020250716001800.png)

- **Deep Q-Networks (DQNs)** are a combination of Q-learning and deep neural networks. Instead of storing Q-values in a table, DQNs use a neural network to approximate the Q-function, allowing the agent to handle complex environments with large or continuous state spaces.

- On the **left**, the DQN takes both a **state** (like a game screen) and an **action** (e.g., “move right”) as input and outputs the predicted Q-value Q(s,a) for that action.
- On the **left side**, for each input, you give the **state** and **one specific action**, so to get Q-values for all actions, you'd need to pass the same state with each possible action **separately**. That means if you have 1 state and **n actions**, you need **n inputs** to the network—one for each (state, action) pair.

- On the **right**, the DQN takes only the **state** and outputs Q-values for **all possible actions**, which is **more efficient** during training.
- The **right side** is better because it allows the network to compute Q-values for **all actions in one forward pass**, given just the state. This is more efficient, especially when the number of actions is large, since we don’t need to run the network separately for each action.

- Uses replay buffer, a target network, and gradient descent to minimize TD error:  
	$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')} \left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right] $$
- This loss function $\mathcal{L}(\theta)$ is used in DQNs to train the Q-network. It minimizes the **temporal difference (TD) error**, which is the difference between the predicted Q-value and the target Q-value. The **replay buffer** stores past experiences to break correlation between samples. A separate **target network** $Q_{\theta^-}$ is used to compute stable target Q-values. **Gradient descent** updates the Q-network $Q_\theta$ to reduce this error over time.
- Isaac: target network weights $\theta^-$ are updated slowly (e.g., every 10k steps).

![Q-Learning Limitation](imgs/Pasted%20image%2020250716002512.png)

- **Q-learning** struggles when the **state or action space is large or continuous** because it needs to store or compute Q-values for every possible state-action pair. In **discrete action spaces**, this works fine if the number of actions is small, but it becomes inefficient or infeasible as the action space grows. Also, Q-learning doesn't naturally handle **continuous actions**, since it relies on computing $\max_a Q(s, a)$, which is hard to do when actions aren’t discrete.

## **5. Policy Learning**

- **Policy learning** is when the agent directly learns a **policy**, a **mapping** from states to actions, *without* relying on a Q-function. Instead of estimating the value of actions and choosing the best, it tries to find the best policy parameters that maximize expected rewards, often using techniques like **policy gradients**. It's especially useful in continuous or high-dimensional action spaces.
- Think of a **policy** as the agent’s **brain or strategy**,it tells the agent **what to do** in each situation.
- More simply:
	If the agent sees a state (like being in a room), the **policy** tells it what action to take (like "go left" or "open door").
    
- There are two types:
	- **Deterministic policy**: always gives the same action for a state (e.g., always "go right").
	- **Stochastic policy**: gives a **probability** for each action (e.g., 70% "go right", 30% "go left").
	
- In reinforcement learning, we try to **improve the policy** so the agent makes better decisions and earns more reward. this can be done using **feedback from rewards** to adjust it, step by step, so it chooses better actions over time. Here's how it's usually done:

	1. **Try the current policy**: Let the agent interact with the environment using its current policy (its decision-making strategy).
	    
	2. **Collect data**: Record the states, actions, and rewards the agent receives.
	    
	3. **Compute gradients**: Use math (calculus) to find out how the policy's decisions affected the total reward.
	    
	4. **Update the policy**: Change the policy a little in the direction that increases expected reward—this is called **gradient ascent**.
    

This is repeated many times, and the policy gradually gets better at choosing actions that lead to higher rewards. This is the idea behind **policy gradient methods**.
## 6. **Policy Gradient Methods (REINFORCE)**

- **Policy gradients** work by directly learning the best way to act in each state, directly optimizing the policy. Instead of using a table or network to choose the best action (like in Q-learning), the policy itself is a neural network that **outputs probabilities of actions** (for discrete actions) or **actual values** (for continuous actions).

- For **continuous actions**, policy gradients are great because the policy can output values from a continuous range—like a speed or angle, rather than choosing from fixed options. The network is trained to adjust these outputs so they lead to higher total rewards, using gradients calculated from experience, it only requires a **mean** and **variance**.

![Policy Gradient Example 1](imgs/Pasted%20image%2020250716003015.png)
![Policy Gradient Example 2](imgs/Pasted%20image%2020250716003023.png)

- Directly optimize policy:  
	$$\nabla_\theta J(\theta) = \mathbb{E} \left[\nabla_\theta \log \pi_\theta(a_t | s_t) R_t\right]$$
- Use **baseline** (e.g., value function) to reduce variance:  
    $$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\,(R_t - b(s_t))\right]$$
## 7. Summary

![Summary Diagram](imgs/Pasted%20image%2020250716003759.png)
    

---

## Example Implementations in `PyTorch`

Let's illustrate a basic REINFORCE algorithm and Q-learning with neural nets:

### 🔹 Deep Q-Network (DQN) in `PyTorch`

Here’s your **DQN (Deep Q-Network)** code with detailed comments explaining each step and how it connects to the lecture:

```python
import random
from collections import deque

# Q-network: approximates the Q-function Q(s, a)
# Outputs Q-values for all possible actions given a state
class QNet(nn.Module):
    def __init__(self, obs_sz, act_sz):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_sz, 128),  # Input: observation size (state), hidden layer
            nn.ReLU(),
            nn.Linear(128, act_sz)   # Output: one Q-value per action
        )
    
    def forward(self, x):
        return self.fc(x)  # Predict Q-values for all actions in current state


# Train a DQN agent using experience replay and a target network
def train_dqn(env_name='CartPole-v1', epochs=200, replay_size=10000, batch_size=64,
              lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=0.995,
              target_update=10):
    
    env = gym.make(env_name)
    obs_sz = env.observation_space.shape[0]  # Size of state
    act_sz = env.action_space.n              # Number of discrete actions

    policy_net = QNet(obs_sz, act_sz)        # Main network (Qθ)
    target_net = QNet(obs_sz, act_sz)        # Target network (Qθ⁻)
    target_net.load_state_dict(policy_net.state_dict())  # Initialize with same weights
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    rb = deque(maxlen=replay_size)  # Experience replay buffer
    eps = eps_start  # Initial epsilon for ε-greedy policy

    for ep in range(epochs):
        state = env.reset()
        done = False
        while not done:
            # ε-greedy action selection: exploration vs. exploitation
            if random.random() < eps:
                action = env.action_space.sample()  # Explore: random action
            else:
                with torch.no_grad():
                    qs = policy_net(torch.tensor(state, dtype=torch.float))  # Get Q-values
                    action = int(qs.argmax().item())  # Exploit: choose best Q(s,a)

            # Interact with environment
            next_state, reward, done, _ = env.step(action)
            # Store experience in replay buffer
            rb.append((state, action, reward, next_state, done))
            state = next_state

            if len(rb) < batch_size:
                continue  # Wait until there's enough data to sample a full batch

            # Sample a random batch from replay buffer
            batch = random.sample(rb, batch_size)
            s, a, r, ns, d = zip(*batch)  # s: states, a: actions, ns: next states, d: done flags

            # Convert batch to tensors
            s = torch.tensor(s, dtype=torch.float)
            a = torch.tensor(a)
            r = torch.tensor(r, dtype=torch.float)
            ns = torch.tensor(ns, dtype=torch.float)
            d = torch.tensor(d, dtype=torch.float)  # 1.0 if done, 0.0 otherwise

            # Compute Q(s, a) from policy_net
            qvals = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()  # Qθ(s,a)
            
            # Compute target Q-value using target_net: r + γ * max_a' Qθ⁻(s', a')
            q_next = target_net(ns).max(1)[0].detach()
            target = r + gamma * q_next * (~d.bool())  # No future reward if done

            # MSE loss between predicted and target Q-values (TD error)
            loss = nn.functional.mse_loss(qvals, target)

            # Backpropagation and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Decay epsilon after each episode to reduce exploration over time
        eps = max(eps_end, eps * eps_decay)

        # Update target network every few episodes
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Log progress
        if ep % 20 == 0:
            print(f"Episode {ep}, eps {eps:.2f}")

    print("Training complete")
```

### How it relates to the lecture:

- **Q-learning** is approximated using a **Q-network (QNet)**.
    
- **Replay buffer** (`deque`) stores past experiences and samples randomly to break correlation and improve training stability.
    
- **Target network** provides stable Q-value targets (used in the loss), preventing oscillations during training.
    
- The **TD error** is minimized using gradient descent, just like in the DQN loss formula from your lecture.
    
- **ε-greedy** policy balances exploration (random action) and exploitation (choose best action from Q-values).

---
### 🔹 REINFORCE in `PyTorch`

```python
import torch, torch.nn as nn, torch.optim as optim
import gym

# This defines the policy network—a neural network that maps states to action probabilities.
# It’s used to represent the policy π(a|s).
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),  # First layer: from input state to 128 hidden units
            nn.ReLU(),                  # Activation to add non-linearity
            nn.Linear(128, action_dim), # Output layer: one value per action
            nn.Softmax(dim=-1)          # Converts raw scores to probabilities over actions
        )
    
    def forward(self, x):
        return self.fc(x)  # Returns the probability distribution over actions for a given state


# Main REINFORCE algorithm: a basic policy gradient method
def reinforce(env_name='CartPole-v1', lr=1e-2, gamma=0.99, episodes=500):
    env = gym.make(env_name)  # Create the environment
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)  # Create policy network
    optimizer = optim.Adam(policy.parameters(), lr=lr)  # Optimizer to adjust policy params

    for ep in range(episodes):  # Repeat for a number of episodes
        state = env.reset()  # Reset environment and get initial state
        log_probs, rewards = [], []  # Store log-probabilities of actions and rewards
        
        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
            probs = policy(state)  # Get action probabilities from policy π(a|s)
            dist = torch.distributions.Categorical(probs)  # Create a categorical distribution over actions
            action = dist.sample()  # Sample an action from the distribution
            log_probs.append(dist.log_prob(action))  # Save log-probability of chosen action
            
            state, r, done, _ = env.step(action.item())  # Take action in the environment
            rewards.append(r)  # Save received reward

        # === Compute discounted returns ===
        # R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        G, returns = 0, []
        for r in rewards[::-1]:  # Loop backwards to compute return for each time step
            G = r + gamma * G
            returns.insert(0, G)  # Prepend to build the return list in the right order
        
        # Normalize returns to reduce variance
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # === Compute loss ===
        # For each step: loss = -logπ(a|s) * G_t (REINFORCE update rule)
        loss = []
        for logp, Gt in zip(log_probs, returns):
            loss.append(-logp * Gt)  # Negative because we want to maximize reward
        
        # Backpropagation step
        optimizer.zero_grad()
        torch.stack(loss).sum().backward()  # Combine all losses and compute gradients
        optimizer.step()  # Update policy parameters (θ) to maximize expected reward

        # Logging every 50 episodes
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, total reward {sum(rewards)}")
```

### Summary of Key Concepts:

- **Policy π(a|s)** is learned directly via a neural network (`PolicyNet`) that outputs a probability distribution over actions.
    
- **Sampling** is done using `Categorical(probs)` to reflect a **stochastic policy**.
    
- **Returns (G)** represent total future reward, used to tell the agent how good its decisions were.
    
- **Loss = -logπ(a|s) * G** follows the **policy gradient** idea: reinforce actions that led to high returns.
    
- **Gradient ascent** is done using PyTorch’s autograd and optimizer to update the policy parameters.
- **`gym`** is a Python library by OpenAI used to create and interact with **reinforcement learning environments**. It provides a simple interface for training and testing RL agents.
---
### `QNet` vs `PolicyNet`

### 🔹 `QNet` (used in DQN)

```python
class QNet(nn.Module):
    ...
    # No softmax
```

- **Purpose**: Approximates the **Q-function** Q(s, a) — it predicts how good each action is in a given state.
    
- **Use case**: Value learning. You **choose** the best action (max Q-value), and use **targets** to train the network using TD learning.
    
- **Output**: Raw Q-values (not probabilities).
    

Think of it as: _“How good is each action in this state?”_ , choose the best one, and learn by reducing prediction error.

---
### 🔹 `PolicyNet` (used in REINFORCE / policy gradients)

```python
class PolicyNet(nn.Module):
    ...
    nn.Softmax(dim=-1)
```

- **Purpose**: Represents the **policy** π(a|s) — it outputs a **probability distribution over actions**, given a state.
    
- **Use case**: Direct policy learning. You sample actions using this distribution and update the policy to increase the chance of good actions (using rewards).
    
- **Output**: A `softmax` vector (probabilities for each action).
    

Think of it as: _“Which action should I try in this state?”_ , learn probabilities and reinforce good ones.

---

### Summary:

| Concept          | `PolicyNet` (Policy Gradient)                  | `QNet` (DQN)                               |
| ---------------- | ---------------------------------------------- | ------------------------------------------ |
| Learns           | policy π(a\|s)                                 | probability of actions                     |
| Output           | Probabilities over all actions (via `Softmax`) | Many Q-values (raw scores, one per action) |
| Action selection | Sample action from distribution                | Take action with max Q                     |
| Training goal    | Maximize expected return                       | Minimize TD error                          |
