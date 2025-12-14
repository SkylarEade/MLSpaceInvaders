import numpy as np
import gymnasium as gym
import ale_py
import cv2


def preprocess_frame(frame):
    """
    Preprocess a game frame: RGB (210 x 160 x 3) -> Grayscale (84 x 84)
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
        self.num_frames = num_frames

    def reset(self, initial_frame):
        """
        Resets the buffer with the initial frame.
        Called at the beginning of each episode.
        """
        self.frames = [initial_frame.copy() for _ in range(self.num_frames)]
        return np.stack(self.frames, axis=0)

    def update(self, new_frame):
        """
        Adds a new frame to the buffer, replacing the oldest frame.
        """
        self.frames = self.frames[1:] + [new_frame.copy()]
        return np.stack(self.frames, axis=0)


class GameStats:
    """
    Generic game statistics tracker for LOGGING ONLY.
    
    IMPORTANT: This class is purely for human-readable monitoring.
    The AI learns ONLY from pixel observations (frames) and scalar rewards.
    This tracking does NOT affect training in any way.
    
    Works with ANY game/environment - no game-specific logic.
    """
    def __init__(self):
        self.reset_episode()
        self.all_episodes = []
        
    def reset_episode(self):
        self.episode_reward = 0
        self.frames = 0
        self.positive_rewards = 0  # Count of positive reward events
        self.negative_rewards = 0  # Count of negative reward events
        self.max_single_reward = 0
        self.actions_taken = {}  # Track action distribution
        
    def update(self, reward, action, info=None):
        """
        Update stats with a single step. 
        Only uses reward, action, and frame count - universally available.
        """
        self.episode_reward += reward
        self.frames += 1
        
        # Track reward events (game-agnostic)
        if reward > 0:
            self.positive_rewards += 1
            self.max_single_reward = max(self.max_single_reward, reward)
        elif reward < 0:
            self.negative_rewards += 1
        
        # Track action distribution
        self.actions_taken[action] = self.actions_taken.get(action, 0) + 1
    
    def end_episode(self):
        """Finalize episode and return stats dict"""
        # Calculate action entropy (measure of action diversity)
        total_actions = sum(self.actions_taken.values())
        action_probs = [count / total_actions for count in self.actions_taken.values()]
        action_entropy = -sum(p * np.log(p + 1e-8) for p in action_probs)
        
        stats = {
            'reward': self.episode_reward,
            'frames': self.frames,
            'positive_events': self.positive_rewards,
            'negative_events': self.negative_rewards,
            'max_reward': self.max_single_reward,
            'action_entropy': action_entropy,
            'fps_equivalent': self.frames,  # Can compute actual FPS externally
        }
        self.all_episodes.append(stats)
        self.reset_episode()
        return stats
    
    def get_recent_stats(self, n=10):
        """Get averaged stats over recent episodes"""
        if not self.all_episodes:
            return None
        recent = self.all_episodes[-n:]
        return {
            'avg_reward': np.mean([e['reward'] for e in recent]),
            'avg_length': np.mean([e['frames'] for e in recent]),
            'avg_positive_events': np.mean([e['positive_events'] for e in recent]),
            'avg_action_entropy': np.mean([e['action_entropy'] for e in recent]),
            'max_reward_seen': max(e['max_reward'] for e in recent),
        }