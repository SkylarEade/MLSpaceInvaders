import numpy as np
import gymnasium as gym
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import time
import pickle
import os

from environment import preprocess_frame, FrameStack, GameStats
from networks import ActorCritic
from ppo import PPO
from rollout_buffer import RolloutBuffer


# ============================================================================
# MODEL SAVE / LOAD
# ============================================================================

def save_model(network, filepath, episode_stats=None, metadata=None):
    """
    Save the trained model to disk.
    
    :param network: ActorCritic network to save
    :param filepath: Path to save (e.g., 'models/ppo_spaceinvaders.pkl')
    :param episode_stats: Optional list of episode statistics
    :param metadata: Optional dict of training metadata
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Collect all weights
    model_data = {
        'cnn': {
            'conv1_weights': network.cnn.conv1_weights,
            'conv1_bias': network.cnn.conv1_bias,
            'conv2_weights': network.cnn.conv2_weights,
            'conv2_bias': network.cnn.conv2_bias,
            'conv3_weights': network.cnn.conv3_weights,
            'conv3_bias': network.cnn.conv3_bias,
        },
        'actor': {
            'fc1_weights': network.actor.fc1_weights,
            'fc1_bias': network.actor.fc1_bias,
            'fc2_weights': network.actor.fc2_weights,
            'fc2_bias': network.actor.fc2_bias,
        },
        'critic': {
            'fc1_weights': network.critic.fc1_weights,
            'fc1_bias': network.critic.fc1_bias,
            'fc2_weights': network.critic.fc2_weights,
            'fc2_bias': network.critic.fc2_bias,
        },
        'episode_stats': episode_stats,
        'metadata': metadata or {},
        'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {filepath}")
    return filepath


def load_model(filepath, network=None):
    """
    Load a trained model from disk.
    
    :param filepath: Path to saved model
    :param network: Optional existing network to load weights into.
                   If None, creates a new ActorCritic.
    :return: (network, episode_stats, metadata)
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    if network is None:
        network = ActorCritic()
    
    # Load CNN weights
    for key, value in model_data['cnn'].items():
        setattr(network.cnn, key, value)
    
    # Load Actor weights
    for key, value in model_data['actor'].items():
        setattr(network.actor, key, value)
    
    # Load Critic weights
    for key, value in model_data['critic'].items():
        setattr(network.critic, key, value)
    
    print(f"Model loaded from: {filepath}")
    print(f"  Saved at: {model_data.get('save_time', 'unknown')}")
    
    return network, model_data.get('episode_stats'), model_data.get('metadata')


def evaluate_agent(network, num_episodes=3, render=True):
    """Evaluate the agent's performance"""
    render_mode = 'human' if render else None
    env = gym.make('ALE/SpaceInvaders-v5', render_mode=render_mode)
    frame_stack = FrameStack(num_frames=4)  # FIXED: was stack_size
    stats = GameStats()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = preprocess_frame(state)
        stacked_state = frame_stack.reset(state)
        stats.reset_episode()
        
        done = False
        print(f"  Playing evaluation episode {episode + 1}...")
        
        while not done:
            action, _, _ = network.get_action_and_value(stacked_state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            stats.update(reward, action, info)
            
            next_state = preprocess_frame(next_state)
            stacked_state = frame_stack.update(next_state)
        
        ep_stats = stats.end_episode()
        episode_rewards.append(ep_stats['reward'])
        print(f"    Episode {episode + 1}: Reward={ep_stats['reward']:.0f}, "
              f"Length={ep_stats['frames']}, Events={ep_stats['positive_events']}, "
              f"ActionEntropy={ep_stats['action_entropy']:.2f}")
    
    env.close()
    return np.mean(episode_rewards)


class ParallelEnvWorker:
    """Worker that collects experience from a single environment"""
    def __init__(self, env_id, render=False):
        self.render_mode = 'human' if render else None
        self.env = gym.make('ALE/SpaceInvaders-v5', render_mode=self.render_mode)
        self.frame_stack = FrameStack(num_frames=4)
        self.stats = GameStats()
        self.reset()
    
    def reset(self):
        state, info = self.env.reset()
        state = preprocess_frame(state)
        self.stacked_state = self.frame_stack.reset(state)
        self.stats.reset_episode()
        return self.stacked_state
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.stats.update(reward, action, info)
        
        # Reward clipping: common practice for Atari to stabilize training
        # Clips rewards to [-1, 1] range while preserving sign
        clipped_reward = np.clip(reward, -1.0, 1.0)
        
        next_state = preprocess_frame(next_state)
        self.stacked_state = self.frame_stack.update(next_state)
        
        ep_stats = None
        if done:
            ep_stats = self.stats.end_episode()
            self.reset()
        
        return self.stacked_state, clipped_reward, done, ep_stats
    
    def close(self):
        self.env.close()


def train_space_invaders(
    total_timesteps=1_000_000,
    n_steps=2048,
    n_epochs=4,
    batch_size=64,
    lr=2.5e-4,
    clip_epsilon=0.1,
    value_coef=0.5,
    entropy_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    eval_freq=10,
    render_eval=True,
    num_envs=1,
    render_training=False,
    save_dir='models',
    save_freq=10,
    model_name='ppo_agent'
):
    """
    Train PPO agent on Space Invaders.
    
    :param total_timesteps: Total environment steps
    :param n_steps: Steps per rollout before update
    :param n_epochs: PPO epochs per update
    :param batch_size: Mini-batch size
    :param lr: Learning rate
    :param clip_epsilon: PPO clip parameter
    :param value_coef: Value loss coefficient
    :param entropy_coef: Entropy bonus coefficient
    :param gamma: Discount factor
    :param gae_lambda: GAE lambda
    :param eval_freq: Evaluate every N updates
    :param render_eval: Whether to render evaluation
    :param num_envs: Number of parallel environments
    :param render_training: Render one environment during training
    :param save_dir: Directory to save models (default: 'models')
    :param save_freq: Save checkpoint every N updates (default: 10)
    :param model_name: Base name for saved models (default: 'ppo_agent')
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("PPO Space Invaders Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Steps per rollout: {n_steps}")
    print(f"Epochs per update: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Clip epsilon: {clip_epsilon}")
    print(f"Num environments: {num_envs}")
    print(f"Render training: {render_training}")
    print(f"Save directory: {save_dir}")
    print(f"Save frequency: every {save_freq} updates")
    print("=" * 60)
    
    # Create environments
    envs = []
    for i in range(num_envs):
        # First env can render if requested
        render = render_training and (i == 0)
        envs.append(ParallelEnvWorker(f'env_{i}', render=render))
    
    # Initialize network and optimizer
    network = ActorCritic()
    ppo = PPO(
        network=network,
        lr=lr,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=0.5
    )
    
    # Buffer stores experiences from all envs
    buffer = RolloutBuffer(
        buffer_size=n_steps * num_envs,
        state_shape=(4, 84, 84),
        gamma=gamma,
        gae_lambda=gae_lambda
    )
    
    # Get initial states
    current_states = [env.stacked_state for env in envs]
    
    # Tracking
    all_episode_stats = []
    episode_count = 0
    update_count = 0
    global_step = 0
    start_time = time.time()
    
    print("\nStarting training...\n")
    
    while global_step < total_timesteps:
        # Collect rollout
        buffer.clear()
        
        for step in range(n_steps):
            # Get actions for all environments
            for env_idx, env in enumerate(envs):
                state = current_states[env_idx]
                action, value, log_prob = network.get_action_and_value(state)
                
                # Step environment
                next_state, reward, done, ep_stats = env.step(action)
                
                # Store transition
                buffer.add(state, action, reward, value, log_prob, float(done))
                
                # Update current state
                current_states[env_idx] = next_state
                
                # Track episode stats
                if ep_stats is not None:
                    all_episode_stats.append(ep_stats)
                    episode_count += 1
                    
                    if episode_count % 5 == 0:
                        recent = all_episode_stats[-10:]
                        avg_reward = np.mean([e['reward'] for e in recent])
                        avg_length = np.mean([e['frames'] for e in recent])
                        avg_entropy = np.mean([e['action_entropy'] for e in recent])
                        print(f"Episode {episode_count}: "
                              f"Reward={avg_reward:.1f}, "
                              f"Length={avg_length:.0f}, "
                              f"ActionEntropy={avg_entropy:.2f}, "
                              f"Steps={global_step:,}")
                
                global_step += 1
                
                if global_step >= total_timesteps:
                    break
            
            if global_step >= total_timesteps:
                break
        
        # Compute advantages using last value estimate
        # Use value from first env's current state as bootstrap
        _, last_value, _ = network.get_action_and_value(current_states[0])
        buffer.compute_advantages(last_value)
        
        # PPO update
        update_count += 1
        print(f"\n{'='*40}")
        print(f"Update {update_count} at step {global_step:,}")
        print(f"{'='*40}")
        
        stats = ppo.update(buffer, n_epochs=n_epochs, batch_size=batch_size)
        
        # Buffer stats
        buf_stats = buffer.get_stats()
        if buf_stats:
            print(f"Buffer: {buf_stats['num_transitions']} transitions, "
                  f"{buf_stats['num_episodes_ended']} episodes ended")
            print(f"Mean value: {buf_stats['mean_value']:.3f}, "
                  f"Mean advantage: {buf_stats['mean_advantage']:.3f}")
        
        # Timing
        elapsed = time.time() - start_time
        fps = global_step / elapsed
        print(f"FPS: {fps:.1f}, Elapsed: {elapsed/60:.1f} min")
        
        # Periodic evaluation
        if render_eval and (update_count % eval_freq == 0):
            print(f"\n{'*'*40}")
            print(f"EVALUATION - Update {update_count}")
            print(f"{'*'*40}")
            avg_eval_reward = evaluate_agent(network, num_episodes=3, render=True)
            print(f"Average Evaluation Reward: {avg_eval_reward:.1f}")
            print(f"{'*'*40}\n")
        
        # Save checkpoint
        if update_count % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_update{update_count}.pkl")
            metadata = {
                'update_count': update_count,
                'global_step': global_step,
                'episode_count': episode_count,
                'total_timesteps': total_timesteps,
                'hyperparameters': {
                    'lr': lr, 'clip_epsilon': clip_epsilon,
                    'value_coef': value_coef, 'entropy_coef': entropy_coef,
                    'gamma': gamma, 'gae_lambda': gae_lambda,
                    'n_steps': n_steps, 'n_epochs': n_epochs, 'batch_size': batch_size
                }
            }
            save_model(network, checkpoint_path, episode_stats=all_episode_stats, metadata=metadata)
        
        print()
    
    # Cleanup
    for env in envs:
        env.close()
    
    # ========== FINAL MODEL SAVE ==========
    final_path = os.path.join(save_dir, f"{model_name}_final.pkl")
    metadata = {
        'update_count': update_count,
        'global_step': global_step,
        'episode_count': episode_count,
        'total_timesteps': total_timesteps,
        'training_time_minutes': (time.time() - start_time) / 60,
        'hyperparameters': {
            'lr': lr, 'clip_epsilon': clip_epsilon,
            'value_coef': value_coef, 'entropy_coef': entropy_coef,
            'gamma': gamma, 'gae_lambda': gae_lambda,
            'n_steps': n_steps, 'n_epochs': n_epochs, 'batch_size': batch_size
        }
    }
    save_model(network, final_path, episode_stats=all_episode_stats, metadata=metadata)
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {global_step:,}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Final model saved to: {final_path}")
    
    if all_episode_stats:
        last_100 = all_episode_stats[-100:]
        print(f"\nLast 100 episodes:")
        print(f"  Avg Reward: {np.mean([e['reward'] for e in last_100]):.1f}")
        print(f"  Avg Episode Length: {np.mean([e['frames'] for e in last_100]):.0f}")
        print(f"  Avg Positive Events: {np.mean([e['positive_events'] for e in last_100]):.1f}")
        print(f"  Avg Action Entropy: {np.mean([e['action_entropy'] for e in last_100]):.2f}")
    
    # Final evaluation
    if render_eval:
        print("\n" + "*" * 40)
        print("FINAL EVALUATION")
        print("*" * 40)
        final_reward = evaluate_agent(network, num_episodes=5, render=True)
        print(f"Final Average Reward: {final_reward:.1f}")
    
    return network, all_episode_stats


def demo_visual_with_background_training():
    """
    Demo showing visual training in foreground while simulating background training.
    One environment renders while others train in background (sequential in this implementation).
    """
    print("=" * 60)
    print("Visual Training Demo")
    print("One environment renders, others train in background")
    print("=" * 60)
    
    # Run training with render_training=True
    # The first environment will render while collecting experience
    network, stats = train_space_invaders(
        total_timesteps=5_000,  # Short demo
        n_steps=256,
        n_epochs=2,
        batch_size=32,
        eval_freq=5,
        render_eval=True,
        num_envs=2,  # 2 environments - first renders
        render_training=True  # Enable visual training
    )
    
    return network, stats


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Space Invaders Training')
    parser.add_argument('--visual', action='store_true', help='Run visual training demo')
    parser.add_argument('--load', type=str, default=None, help='Path to model to load and continue training')
    parser.add_argument('--play', type=str, default=None, help='Path to model to load and play (no training)')
    parser.add_argument('--timesteps', type=int, default=100_000, help='Total training timesteps')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--save-freq', type=int, default=10, help='Save every N updates')
    parser.add_argument('--name', type=str, default='ppo_agent', help='Model name for saving')
    
    args = parser.parse_args()
    
    if args.play:
        # Load and play mode (no training)
        print(f"Loading model from {args.play} for evaluation...")
        network, stats, metadata = load_model(args.play)
        print("\nPlaying 5 episodes...")
        evaluate_agent(network, num_episodes=5, render=True)
        
    elif args.visual:
        # Visual demo mode
        demo_visual_with_background_training()
        
    else:
        # Normal training
        train_space_invaders(
            total_timesteps=args.timesteps,
            n_steps=2048,          
            n_epochs=4,            
            batch_size=256,        
            lr=1e-4,               
            clip_epsilon=0.2,      
            value_coef=0.5,
            entropy_coef=0.01,
            eval_freq=5,
            render_eval=True,
            num_envs=1,
            render_training=False,
            save_dir=args.save_dir,
            save_freq=args.save_freq,
            model_name=args.name
        )