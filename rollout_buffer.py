import numpy as np


class RolloutBuffer:
    """
    Stores experiences for PPO training.
    Implements Generalized Advantage Estimation (GAE).
    """
    def __init__(self, buffer_size, state_shape, gamma, gae_lambda):
        """
        :param buffer_size: Number of transitions to store
        :param state_shape: Shape of state observations
        :param gamma: Discount factor (e.g., 0.99)
        :param gae_lambda: GAE lambda - higher values trust rollouts more (e.g., 0.95)
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Pre-allocate arrays
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.position = 0

    def add(self, state, action, reward, value, log_prob, done):
        """Add a single transition to the buffer"""
        if self.position >= self.buffer_size:
            raise ValueError(f"Buffer overflow: position {self.position} >= size {self.buffer_size}")
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.dones[self.position] = done
        self.position += 1

    def compute_advantages(self, last_value):
        """
        Compute GAE advantages and returns.
        
        GAE: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        :param last_value: V(s_T) - value of the state after the last collected transition
        """
        last_advantage = 0
        
        for t in reversed(range(self.position)):
            # Determine next value
            if t == self.position - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
            self.advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = self.advantages[t]
        
        # Returns = Advantages + Values (for value function target)
        self.returns[:self.position] = self.advantages[:self.position] + self.values[:self.position]

    def get(self):
        """Return all collected data as a dictionary"""
        return {
            'states': self.states[:self.position],
            'actions': self.actions[:self.position],
            'rewards': self.rewards[:self.position],
            'values': self.values[:self.position],
            'log_probs': self.log_probs[:self.position],
            'dones': self.dones[:self.position],
            'advantages': self.advantages[:self.position],
            'returns': self.returns[:self.position]
        }

    def clear(self):
        """Reset buffer for next rollout collection"""
        self.position = 0
        # Note: No need to zero arrays - they'll be overwritten

    def get_stats(self):
        """Get statistics about the current buffer contents"""
        if self.position == 0:
            return None
        return {
            'mean_reward': np.mean(self.rewards[:self.position]),
            'mean_value': np.mean(self.values[:self.position]),
            'mean_advantage': np.mean(self.advantages[:self.position]),
            'std_advantage': np.std(self.advantages[:self.position]),
            'num_transitions': self.position,
            'num_episodes_ended': int(np.sum(self.dones[:self.position]))
        }