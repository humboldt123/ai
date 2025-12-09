import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable
from collections import deque
import PIL
import matplotlib.pyplot as plt
import json
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
tf.random.set_seed(42)
np.random.seed(42)

import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import ale_py

gym.register_envs(ale_py)

max_episode_steps = 27000
environment_name = "ALE/Breakout-v5"

class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        super().step(1)
        return obs, info
    
    def step(self, action):
        lives_before_action = self.env.unwrapped.ale.lives()
        obs, reward, terminated, truncated, info = super().step(action)
        if self.env.unwrapped.ale.lives() < lives_before_action and not (terminated or truncated):
            super().step(1)
        return obs, reward, terminated, truncated, info

def create_env(render_mode=None):
    env = gym.make(environment_name, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessingWithAutoFire(env, frame_skip=4)
    env = FrameStackObservation(env, stack_size=4)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

env = create_env()

input_shape = (84, 84, 4)
n_actions = env.action_space.n

@register_keras_serializable()
class Normalize(keras.layers.Layer):
    """Normalize uint8 images to [0, 1] float32"""
    def call(self, inputs):
        return tf.cast(inputs, tf.float32) / 255.0
    
    def get_config(self):
        return super().get_config()
    
def create_q_network():
    return keras.Sequential([
        keras.layers.Input(shape=input_shape),
        Normalize(),  # keras.layers.Lambda(lambda obs: tf.cast(obs, tf.float32) / 255.),
        keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'),
        keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
        keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(n_actions)
    ])

q_net = create_q_network()
target_q_net = create_q_network()
target_q_net.set_weights(q_net.get_weights())

train_step_counter = tf.Variable(0)
update_period = 4
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=250000 // update_period,
    end_learning_rate=0.01)

gamma = 0.99
target_update_period = 2000
loss_fn = keras.losses.Huber(reduction="none")

class ReplayBuffer:
    def __init__(self, max_length=100000):
        self.buffer = deque(maxlen=max_length)
    
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_obs = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(max_length=100000)

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    
    def __call__(self, done):
        self.counter += 1
        if self.counter % 100 == 0:
            print(f"\r{self.counter}/{self.total}", end="")

class TrainMetrics:
    def __init__(self):
        self.num_episodes = 0
        self.num_steps = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.current_episode_return = 0
        self.current_episode_length = 0
    
    def step(self, reward, done):
        self.num_steps += 1
        self.current_episode_return += reward
        self.current_episode_length += 1
        
        if done:
            self.num_episodes += 1
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_return = 0
            self.current_episode_length = 0
    
    def average_return(self, last_n=100):
        if len(self.episode_returns) == 0:
            return 0.0
        return np.mean(self.episode_returns[-last_n:])
    
    def average_length(self, last_n=100):
        if len(self.episode_lengths) == 0:
            return 0.0
        return np.mean(self.episode_lengths[-last_n:])

train_metrics = TrainMetrics()

def log_metrics(metrics):
    print(f"\nNumberOfEpisodes = {metrics.num_episodes}")
    print(f"EnvironmentSteps = {metrics.num_steps}")
    print(f"AverageReturn = {metrics.average_return():.1f}")
    print(f"AverageEpisodeLength = {metrics.average_length():.1f}")

class CollectDriver:
    def __init__(self, env):
        self.env = env
        self.obs, _ = env.reset()
    
    def run(self, q_net, replay_buffer, metrics, epsilon, num_steps):
        for _ in range(num_steps):
            obs_t = np.transpose(self.obs, (1, 2, 0))
            
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                q_values = q_net(obs_t[np.newaxis], training=False)
                action = tf.argmax(q_values[0]).numpy()
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            next_obs_t = np.transpose(next_obs, (1, 2, 0))
            replay_buffer.add(obs_t, action, reward, next_obs_t, done)
            
            metrics.step(reward, done)
            
            if done:
                self.obs, _ = self.env.reset()
            else:
                self.obs = next_obs

collect_driver = CollectDriver(env)

@tf.function
def dqn_train_step(obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch):
    obs_batch = tf.cast(obs_batch, tf.float32)
    next_obs_batch = tf.cast(next_obs_batch, tf.float32)
    actions_batch = tf.cast(actions_batch, tf.int32)
    rewards_batch = tf.cast(rewards_batch, tf.float32)
    dones_batch = tf.cast(dones_batch, tf.float32)
    
    with tf.GradientTape() as tape:
        q_values = q_net(obs_batch, training=True)
        batch_indices = tf.range(tf.shape(actions_batch)[0])
        action_indices = tf.stack([batch_indices, actions_batch], axis=1)
        q_values = tf.gather_nd(q_values, action_indices)
        
        next_q_values = target_q_net(next_obs_batch, training=False)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        
        target_q_values = rewards_batch + gamma * max_next_q_values * (1.0 - dones_batch)
        
        loss = loss_fn(target_q_values, q_values)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, q_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_net.trainable_variables))
    
    return loss

def evaluate_policy(q_net, epsilon, num_episodes=10):
    """Evaluate policy with epsilon-greedy and return average discounted return"""
    eval_env = create_env()
    discounted_returns = []
    
    for ep in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        discounted_return = 0
        discount = 1.0
        
        while not done:
            obs_t = np.transpose(obs, (1, 2, 0))
            
            if np.random.rand() < epsilon:
                action = eval_env.action_space.sample()
            else:
                q_values = q_net(obs_t[np.newaxis], training=False)
                action = tf.argmax(q_values[0]).numpy()
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            discounted_return += discount * reward
            discount *= gamma
        
        discounted_returns.append(discounted_return)
    
    eval_env.close()
    return np.mean(discounted_returns)

def save_checkpoint(q_net, train_step, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    epsilon = epsilon_fn(train_step).numpy()
    filename = f"dqn_step{train_step}_eps{epsilon:.3f}.keras"
    filepath = os.path.join(checkpoint_dir, filename)
    q_net.save(filepath)
    print(f"\nSaved checkpoint: {filepath}")
    return filepath

def load_checkpoint(filepath):
    """Load model and extract epsilon from filename"""
    # # HACK: MY BAD!!!! I SHOULD HAVE USED Normal layer instead of lambda OOPS
    # import tensorflow as tf
    # keras.config.enable_unsafe_deserialization() 
    # #T ^ THATS THE FIX
    q_net = keras.models.load_model(filepath)
    filename = os.path.basename(filepath)
    epsilon_str = filename.split("_eps")[1].split(".keras")[0]
    epsilon = float(epsilon_str)
    return q_net, epsilon

def save_checkpoint_metadata(checkpoint_data, filepath="checkpoint_results.json"):
    """Save checkpoint evaluation results to JSON"""
    metadata = [
        {"step": int(step), "epsilon": float(epsilon), "avg_return": float(avg_return)}
        for step, epsilon, avg_return in checkpoint_data
    ]
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved checkpoint metadata to {filepath}")

def load_checkpoint_metadata(filepath="checkpoint_results.json"):
    """Load checkpoint metadata from JSON"""
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    return [(d["step"], d["epsilon"], d["avg_return"]) for d in metadata]

def plot_training_curve_from_metadata(filepath="checkpoint_results.json", output="trainingCurve.png"):
    """Recreate training curve from saved metadata"""
    data = load_checkpoint_metadata(filepath)
    steps = [d[0] for d in data]
    returns = [d[2] for d in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, returns, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Average Discounted Return', fontsize=12)
    plt.title('DQN Training Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved training curve to {output}")
    plt.close()

def evaluate_all_checkpoints(checkpoint_dir="checkpoints", num_episodes=10):
    """Evaluate all saved checkpoints in directory"""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "dqn_step*.keras")))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []
    
    results = []
    for checkpoint_path in checkpoint_files:
        print(f"Evaluating {checkpoint_path}...")
        q_net_loaded, epsilon = load_checkpoint(checkpoint_path)
        
        filename = os.path.basename(checkpoint_path)
        step = int(filename.split("_step")[1].split("_eps")[0])
        
        avg_return = evaluate_policy(q_net_loaded, epsilon, num_episodes)
        results.append((step, epsilon, avg_return))
        print(f"  Step {step}, ε={epsilon:.3f}, Avg Return={avg_return:.2f}")
    
    return results

def train_agent_with_checkpoints(n_iterations, checkpoint_interval=None):
    """Train with checkpoints every checkpoint_interval iterations"""
    if checkpoint_interval is None:
        checkpoint_interval = n_iterations // 10
    
    checkpoint_data = []
    
    for iteration in range(n_iterations):
        epsilon = epsilon_fn(train_step_counter.numpy())
        collect_driver.run(q_net, replay_buffer, train_metrics, epsilon, update_period)
        
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = replay_buffer.sample(64)
        loss = dqn_train_step(obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch)
        
        train_step_counter.assign_add(1)
        
        if train_step_counter.numpy() % target_update_period == 0:
            target_q_net.set_weights(q_net.get_weights())
        
        print(f"\r{iteration}/{n_iterations} loss:{loss.numpy():.5f}", end="")
        
        if (iteration + 1) % checkpoint_interval == 0:
            current_step = train_step_counter.numpy()
            current_epsilon = epsilon_fn(current_step).numpy()
            
            print(f"\n\nCheckpoint at iteration {iteration + 1}...")
            save_checkpoint(q_net, current_step)
            
            print(f"Evaluating with epsilon={current_epsilon:.3f}...")
            avg_return = evaluate_policy(q_net, current_epsilon, num_episodes=10)
            checkpoint_data.append((current_step, current_epsilon, avg_return))
            print(f"Average discounted return: {avg_return:.2f}\n")
            
            log_metrics(train_metrics)
    
    if checkpoint_data:
        save_checkpoint_metadata(checkpoint_data)
        
        steps = [d[0] for d in checkpoint_data]
        returns = [d[2] for d in checkpoint_data]
        plt.figure(figsize=(10, 6))
        plt.plot(steps, returns, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Average Discounted Return', fontsize=12)
        plt.title('DQN Training Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig('trainingCurve.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved training curve to trainingCurve.png")
        plt.close()
    
    return checkpoint_data

def play_and_record(checkpoint_path, output_path="gameplay.gif", num_frames=1000):
    """Load checkpoint, play game with epsilon-greedy, record video"""
    q_net_loaded, epsilon = load_checkpoint(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    print(f"Using epsilon={epsilon:.3f}")
    
    eval_env = create_env(render_mode="rgb_array")
    
    frames = []
    obs, _ = eval_env.reset()
    
    for step in range(num_frames):
        frames.append(eval_env.render())
        
        obs_t = np.transpose(obs, (1, 2, 0))
        
        if np.random.rand() < epsilon:
            action = eval_env.action_space.sample()
        else:
            q_values = q_net_loaded(obs_t[np.newaxis], training=False)
            action = tf.argmax(q_values[0]).numpy()
        
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        
        if done:
            obs, _ = eval_env.reset()
    
    eval_env.close()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    frame_images = [PIL.Image.fromarray(frame) for frame in frames[:min(150, len(frames))]]
    frame_images[0].save(output_path, format='GIF',
                         append_images=frame_images[1:],
                         save_all=True,
                         duration=30,
                         loop=0)
    print(f"Saved gameplay video to {output_path}")

def warmup_replay_buffer(num_steps=20000):
    """Fill replay buffer with random experiences"""
    print("Warming up replay buffer...")
    show_progress = ShowProgress(num_steps)
    for step in range(num_steps):
        obs_t = np.transpose(collect_driver.obs, (1, 2, 0))
        action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        next_obs_t = np.transpose(next_obs, (1, 2, 0))
        replay_buffer.add(obs_t, action, reward, next_obs_t, done)
        
        train_metrics.step(reward, done)
        show_progress(done)
        
        if done:
            collect_driver.obs, _ = env.reset()
        else:
            collect_driver.obs = next_obs
    print("\nWarmup complete!")

if __name__ == "__main__":
    # Warmup
    warmup_replay_buffer(20000)
    
    # Train with checkpoints
    print("\nStarting training...")
    checkpoint_data = train_agent_with_checkpoints(n_iterations=500000)
    
    # Example: Re-evaluate all checkpoints later
    results = evaluate_all_checkpoints()
    save_checkpoint_metadata(results, "re_evaluated_results.json")
    plot_training_curve_from_metadata("re_evaluated_results.json", "new_curve.png")
    
    # Example: Load and play a specific checkpoint
    #play_and_record("checkpoints/dqn_step500000_eps0.010.keras", "gameplay_step500000.gif")