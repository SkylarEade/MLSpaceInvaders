import numpy as np
import gymnasium as gym
import ale_py
import cv2


def preprocess_frame(frame):
    """
    Used to preprocess a game frame, taking in a frame as a large a RGB array (210 x 160 x 3),
    returning it as a compressed 84 x 84 grayscale array
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame

class FrameStack:
    """
    Maintains a stack of N frames and returns that stack in a (N, 84, 84) array.
    """
    def __init__(self, num_frames=4):
        self.frames = []
        self.num_frames = num_frames # total stored frames

    def reset(self, initial_frame):
        """
        Resets the buffer with the initial frame 
        Called at the beginning of each episode
        """
        self.frames = []
        for _ in range(self.num_frames):
            self.frames.append(initial_frame)
        return np.stack(self.frames, axis=0)

    def update(self, new_frame):
        """
        Adds a new frame to the buffer, replacing the oldest buffer frame
        """
        self.frames = self.frames[1:] + [new_frame]
        return np.stack(self.frames, axis=0)


# Load environment
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')

env.close()