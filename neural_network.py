import numpy as np
import gymnasium as gym
import ale_py
import torch
import cv2



class Actor():
    """Takes in the current environment (state) and determines the best action to take from there"""
    def __init__(self):
        pass

    def forward_prop(self, X):
        pass

    def back_prop(self, X):
        pass

    def update_weights(self):
        pass


class Critic():
    """Evaluates the action based on the environment state and returns a score for that action"""
    def __init__(self):
        pass

    def forward_prop(self, X):
        pass

    def back_prop(self, X):
        pass

    def update_weights(self):
        pass

class NetworkHandler():
