## Sources:
- [How to Use Learning Rate Scheduling for Neural Network Training](https://www.youtube.com/watch?v=4FcW7OkIZLw)
- [PyTorch LR Scheduler - Adjust The Learning Rate For Better Results](https://www.youtube.com/watch?v=81NJgoR5RfY)

---
## Learning Rate Schedules

The **learning rate (LR)** controls how big a step the optimizer takes during each update.
A **learning rate schedule** *changes* the LR over time to improve training stability and convergence.

---

### **1. Step Decay**

* **Idea:** Reduce the learning rate by a fixed factor after a set number of epochs.
* **Formula:**
  $$
  \eta_t = \eta_0 \times \gamma^{\lfloor t / s \rfloor}
  $$
  where ( s ) = step size, ( \gamma ) = decay factor.
* **Example:** Reduce LR by 0.1 every 30 epochs.
* **In PyTorch:**
  `torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)`

> Think of it as: “Drops LR suddenly at intervals → simple but effective.”

---

### **2. Exponential Decay**

* **Idea:** Continuously decrease the LR exponentially with each step or epoch.
* **Formula:**
  $$
  \eta_t = \eta_0 \times e^{-kt}
  $$
  where ( k ) is a decay constant.
* **Use case:** When you want a **smooth** LR decrease instead of sharp drops.
* **In PyTorch:**
  `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)`

> Think of it as: “Gradually slows down learning to fine-tune at the end.”

---

### **3. Cosine Annealing**

* **Idea:** The LR follows a cosine curve: starts high, then slowly decreases to near zero.
* **Formula:**
  $$
  \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T_{max}}\pi))
  $$
* **Use case:** Popular in modern training (e.g., transformers, CNNs).
* **In PyTorch:**
  `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)`

> Think of it as: “High LR at first → smooth cool-down.”

---

### **4. Cyclical LR (CLR)**

* **Idea:** LR cyclically *increases* and *decreases* between two bounds.
* **Why:** Helps escape local minima and improves generalization.
* **In PyTorch:**
  `torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000)`

> Think of it as: “Let the LR breathe → goes up and down during training.”

---

### **5. One Cycle Policy**

* **Idea:** LR starts low, **increases to a max**, then **decreases to near zero**.
* **Why:** Encourages faster convergence and better minima (used in fastai, etc.).
* **In PyTorch:**
  `torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps)`

> Think of it as: “A single rise and fall → aggressive yet stable.”

---

### **Quick Summary Table**

| Schedule Type         | LR Pattern                 | Best For                | Key Intuition                       |
| --------------------- | -------------------------- | ----------------------- | ----------------------------------- |
| **Step Decay**        | Drops in steps             | Simple CNN/MLP training | Sudden drops after milestones       |
| **Exponential Decay** | Smooth continuous decrease | Long training runs      | Gradual slowdown                    |
| **Cosine Annealing**  | Cosine-shaped decrease     | Modern deep nets        | Smooth cooldown                     |
| **Cyclical LR**       | Oscillates between bounds  | Escaping local minima   | Periodic exploration                |
| **One Cycle Policy**  | Rise → peak → fall         | Fast training           | Quick exploration, fine convergence |
