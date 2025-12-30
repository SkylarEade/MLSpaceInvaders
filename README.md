# Machine Learning Space Invaders AI
**Reinforcement learning project for understanding PPO (Proximal Policy Optimization) made completely from scratch using only numpy**

### What it Does
Trains and saves a neural network that learns to play Space Invaders purely from visual input (pixels) using reinforcement learning

### Lessons Learned
- Reinforcement learning fundamentals (states, actions, rewards, episodes)
- Convolutional Neural Networks for visual feature extraction
- Actor-Critic architecture (separate policy and value networks)
- Forward and backward propagation through CNN layers
- Proximal Policy Optimization (PPO) algorithm
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Entropy bonus for exploration vs exploitation balance
- Frame stacking for temporal information
- Reward clipping for training stability
- Adam optimizer with gradient clipping

### How to Run
```
git clone https://github.com/SkylarEade/MLSpaceInvaders.git
cd MLSpaceInvaders
pip install -r requirements.txt
```
To train run:
```
python main.py
```
To train with custom settings:
```
python main.py --timesteps 500000 --save-dir models --name my_agent
```
To watch a trained model play:
```
python main.py --play models/ppo_agent_final.pkl
```
To watch training in real-time (visual mode):
```
python main.py --visual
```

### Command Line Options
| Flag | Description | Default |
|------|-------------|---------|
| `--timesteps` | Total training steps | 100,000 |
| `--save-dir` | Directory to save models | models |
| `--save-freq` | Save checkpoint every N updates | 10 |
| `--name` | Model name for saving | ppo_agent |
| `--play` | Path to model to watch play | None |
| `--visual` | Run visual training demo | False |
