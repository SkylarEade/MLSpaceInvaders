import numpy as np
import gymnasium as gym
import ale_py
import torch
import cv2

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
class NeuralNetwork():
    def __init__(self):
        pass

    def forward_prop(self, X):
        pass

    def calculate_loss(self, y_one_hot):
        pass

    def back_prop(self, X, y_one_hot):
        pass

    def update_weights(self):
        pass
