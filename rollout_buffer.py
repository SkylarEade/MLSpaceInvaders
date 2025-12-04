import numpy as np

class RolloutBuffer:
    """
    Stores past experiences for Proximal Policy Optimization
    """
    def __init__(self, buffer_size, state_shape, gamma, gae_lambda):
        """
        Docstring for __init__
        
        :param buffer_size: How many transitions to store
        :param state_shape: Shape of states
        :param gamma: Discount factor 
        :param gae_lambda: Generalized Advantage Estimation lambda > Larger lambda trusts rollouts more, smaller trusts critic more
        """
        self.states = np.zeros((buffer_size, *state_shape))
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, np.float32)
        self.advantages = np.zeros(buffer_size)
        self.returns = np.zeros(buffer_size)
        self.position = 0
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def add(self, state, action, reward, value, log_prob, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.dones[self.position] = done
        self.position += 1

    def compute_advantages(self, last_value):
        last_advantage = 0 # No future advantages at the end
        for t in reversed(range(self.position)): # loop through transitions (backwards)
            if t == self.position - 1:
                next_value = last_value
            else:
                next_value = self.values[t+1]

            next_value = next_value * (1 - self.dones[t])

            td_err = self.rewards[t] + self.gamma * next_value - self.values[t] # Calculate Temporal Difference error
            self.advantages[t] = td_err + self.gamma * self.gae_lambda * (1-self.dones[t]) * last_advantage # Calculate GAE
            last_advantage = self.advantages[t]
        self.returns[:self.position] = self.advantages[:self.position] + self.values[:self.position]

    def get(self):
        """Returns all data for training"""
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
        self.position = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.values.fill(0)
        self.log_probs.fill(0)
        self.dones.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)