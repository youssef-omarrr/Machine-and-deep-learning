###  1. **CartPole (Starter Project)**

- **Goal**: Keep the pole balanced by moving left or right.
    
- **Why**: Simple, well-known, fast to train.
    
- **Library**: `gym` (`CartPole-v1`)
    

---

###  2. **Brick Breaker Game (Breakout)**

- **Goal**: Control a paddle to bounce a ball and break bricks.
    
- **Why**: Visual and satisfying; clear reward (hit = +1).
    
- **Library**: `gym` (`ALE/Breakout-v5` or via `gym-retro`)
    
- **Challenge**: Uses pixel input (image-based RL), so you might use **Deep Q Networks + CNN**.
    

---

###  3. **Lunar Lander**

- **Goal**: Land a spacecraft gently between flags.
    
- **Why**: Physics-based and satisfying to watch.
    
- **Library**: `gym` (`LunarLander-v2`)
    
- **Techniques**: Great for trying both **DQN** and **policy gradient**.
    

---
### 4. **Protein Motif Completion**

- **Goal**: Predict the full protein sequence from a **partial input** containing a known **biological motif** (e.g., "RGD---").
    
- **Why**: Mimics **biological prompting** as done in EvoDiff, letting you generate or complete biologically meaningful protein sequences.
    
- **Libraries**:
    
    - `PyTorch` – for building and training the model
        
    - `Bio` (from `biopython`) – to parse protein FASTA files
        
    - `scikit-learn` or `NumPy` – for splitting and managing data
        
    - `tqdm` – for progress bars while training
        
    - _(Optional)_ `transformers` – if you want to use pre-trained models like ESM
        
- **Techniques**:
    
    - **Tokenization** of amino acid sequences (one-hot or integer)
        
    - **Masked language modeling** or **sequence-to-sequence learning**
        
    - **Transformer Encoder** or **LSTM** as the model backbone
        
    - Train on real sequences, randomly mask some residues, and learn to fill them in
        

---