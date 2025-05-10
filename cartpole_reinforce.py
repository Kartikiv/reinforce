import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import copy
'''
1. Model is Stupid 
2. Does not know when it is doing to much or doing too little
3. Dumb Forgets what ever good behaviour it learned over time and sometimes becomes an overfit
4. Gets stuck at a high score at stays there 
5. does not understand importance of the angle of the pole and thinks like a moron trying to hold on to the pole while it slides
6. Do something to fix this minor issues


'''

# Configuration
EPISODES = 4000
RENDER_MODE = None  # Set to "human" for visualization
MAX_STEPS = 1500  # Target score
GAMMA = 0.99
LEARNING_RATE = 0.0003  # Lower learning rate for stability

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Ensure output directories exist
os.makedirs("./save_model", exist_ok=True)
os.makedirs("./save_graph", exist_ok=True)


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm"""

    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()

        # Simple network - CartPole is simple enough
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

        self._init_weights()

    def _init_weights(self):
        # Careful initialization to help convergence
        for m in [self.fc1, self.fc2, self.fc3]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)


class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Create policy network
        self.policy = PolicyNetwork(state_size, action_size)

        # Create a target network that will be updated less frequently
        # This helps prevent forgetting good policies
        self.target_policy = copy.deepcopy(self.policy)

        # Lower learning rate for stability
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )

        # Gentle learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.9,  # Very gentle reduction
            patience=300,
            threshold=5.0,
            min_lr=1e-5
        )

        # Memory for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

        # Store multiple good episodes completely
        self.good_episodes = []
        self.max_good_episodes = 5

        # Performance tracking
        self.best_score = 0
        self.best_avg_score = 0
        self.plateau_counter = 0

        # State normalization (adaptive)
        self.state_mean = np.zeros(state_size)
        self.state_std = np.ones(state_size)
        self.update_count = 0

        # Conservatism factor - helps prevent forgetting
        self.conservatism = 0.0  # Start with no conservatism, increase as needed

    def normalize_state(self, state):
        """Use fixed normalization based on CartPole state ranges"""
        # Ensure state_mean and state_std are numpy arrays
        if not isinstance(self.state_mean, np.ndarray):
            self.state_mean = np.array(self.state_mean, dtype=np.float32)
        if not isinstance(self.state_std, np.ndarray):
            self.state_std = np.array(self.state_std, dtype=np.float32)

        # Blend between learned normalization and fixed normalization
        # Fixed normalization constants based on CartPole's typical ranges
        fixed_mean = np.array([0.0, 0.0, 0.0, 0.0])
        fixed_std = np.array([2.4, 2.0, 0.2, 2.0])  # Position, velocity, angle, angle velocity

        # Update running statistics (only during early training)
        if self.update_count < 10000:
            self.update_count += 1
            alpha = 0.01  # Slow update rate
            delta = state[0] - self.state_mean
            self.state_mean = self.state_mean + alpha * delta
            self.state_std = (1 - alpha) * self.state_std + alpha * np.abs(delta)

        # Blend learned and fixed normalization (more fixed normalization over time)
        blend_factor = min(0.8, self.update_count / 10000)  # How much to use fixed normalization
        norm_mean = (1 - blend_factor) * self.state_mean + blend_factor * fixed_mean
        norm_std = (1 - blend_factor) * self.state_std + blend_factor * fixed_std

        # Normalize state
        normalized = (state - norm_mean) / (norm_std + 1e-8)
        return np.clip(normalized, -5, 5)

    def select_action(self, state, training=True):
        """Select action using current policy"""
        # Add noise to state during training for robustness
        if training and self.update_count > 5000:  # After initial learning phase
            noise = np.random.normal(0, 0.01, size=state.shape)
            state = state + noise

        # Normalize state
        norm_state = self.normalize_state(state)
        state_tensor = torch.FloatTensor(norm_state)

        # During training, blend current and target policies
        if training and self.conservatism > 0:
            # Get action probabilities from both networks
            with torch.no_grad():
                target_probs = self.target_policy(state_tensor)

            # Get action probabilities from current policy
            current_probs = self.policy(state_tensor)

            # Blend probabilities based on conservatism factor
            action_probs = (1 - self.conservatism) * current_probs + self.conservatism * target_probs
        else:
            # Just use current policy
            action_probs = self.policy(state_tensor)

        if training:
            # Sampling with temperature
            if self.plateau_counter > 200:
                # Increase exploration when stuck
                temperature = 1.2
                action_probs = action_probs ** (1 / temperature)
                action_probs = action_probs / action_probs.sum()

            # Sample action
            dist = Categorical(action_probs)
            action = dist.sample()

            # Store log probability for training
            self.log_probs.append(dist.log_prob(action))

            # Occasionally force exploration on plateau
            if self.plateau_counter > 300 and random.random() < 0.1:
                # Pick the less likely action
                if action_probs[0] > action_probs[1]:
                    action = torch.tensor(1)
                else:
                    action = torch.tensor(0)
                # Update log prob
                self.log_probs[-1] = dist.log_prob(action)

            return action.item()
        else:
            # During evaluation, always select best action
            return torch.argmax(action_probs).item()

    def store_transition(self, state, action, reward):
        """Store transition for training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def store_good_episode(self):
        """Store a complete good episode"""
        if len(self.rewards) > 100:  # Only store reasonably good episodes
            episode = {
                'states': self.states.copy(),
                'actions': self.actions.copy(),
                'rewards': self.rewards.copy(),
                'total_reward': sum(self.rewards),
                'length': len(self.rewards)
            }

            # Add to good episodes, maintaining only the best ones
            self.good_episodes.append(episode)
            self.good_episodes.sort(key=lambda x: x['total_reward'], reverse=True)

            if len(self.good_episodes) > self.max_good_episodes:
                self.good_episodes.pop()  # Remove worst episode

            print(f"Stored good episode with score {episode['total_reward']}")

    def compute_returns(self):
        """Calculate discounted returns"""
        returns = []
        R = 0

        # Calculate discounted rewards
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        # Convert to tensor
        returns = torch.FloatTensor(returns)

        # Normalize returns for stable training (if we have enough)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update_policy(self, score):
        """Update policy network with conservatism"""
        if len(self.rewards) == 0:
            return 0

        # Store good episodes for replay
        if score > 800 or (score > 500 and len(self.good_episodes) < 2):
            self.store_good_episode()

            # Update target network when we get a really good episode
            # This helps preserve good policies
            if score > 900:
                print(f"Updating target network with score {score}")
                self.target_policy.load_state_dict(self.policy.state_dict())

        # Calculate returns
        returns = self.compute_returns()

        # Policy loss: negative log probability * return
        policy_loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss += -log_prob * R

        # Add entropy regularization to encourage exploration
        if len(self.states) > 0:
            normalized_states = [self.normalize_state(s) for s in self.states]
            state_tensor = torch.FloatTensor(normalized_states)
            action_probs = self.policy(state_tensor)

            # Calculate entropy: -sum(p_i * log(p_i))
            entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()

            # Add entropy bonus to loss - higher coefficient means more exploration
            entropy_coef = 0.01
            policy_loss = policy_loss - entropy_coef * entropy

        # Add conservatism loss if enabled (to prevent forgetting)
        conservatism_loss = 0
        if self.conservatism > 0 and self.best_score > 800:
            # Sample states from the current episode
            sample_size = min(50, len(self.states))
            indices = np.random.choice(len(self.states), sample_size, replace=False)

            # Get states from both sample
            sample_states = [self.states[i] for i in indices]
            normalized_states = [self.normalize_state(s) for s in sample_states]
            state_tensor = torch.FloatTensor(normalized_states)

            # Get action probs from both networks
            with torch.no_grad():
                target_probs = self.target_policy(state_tensor)

            current_probs = self.policy(state_tensor)

            # KL divergence loss to stay close to target
            kl_div = (target_probs * torch.log(target_probs / (current_probs + 1e-10))).sum(dim=1).mean()
            conservatism_loss = self.conservatism * kl_div

            policy_loss += conservatism_loss

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()

        # Gradient clipping (conservative)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

        # Update weights
        self.optimizer.step()

        # Update learning rate based on performance
        self.scheduler.step(score)

        # Adjust conservatism based on performance
        # Increase conservatism if we start forgetting good policies
        if score < self.best_score - 100 and self.best_score > 800:
            # We're forgetting good policies, increase conservatism
            self.conservatism = min(0.3, self.conservatism + 0.02)
            print(f"Increasing conservatism to {self.conservatism:.2f}")
        elif score > self.best_score - 50:
            # We're doing well, decrease conservatism
            self.conservatism = max(0.0, self.conservatism - 0.005)

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

        return policy_loss.item()

    def replay_good_episode(self):
        """Replay a good episode to reinforce successful behavior"""
        if not self.good_episodes:
            return 0

        # Choose one of the best episodes at random (with preference for better ones)
        weights = [ep['total_reward'] for ep in self.good_episodes]
        episode = random.choices(self.good_episodes, weights=weights)[0]

        # Prepare for policy update
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

        # Process the good episode
        for i in range(len(episode['states'])):
            state = episode['states'][i]
            action = episode['actions'][i]
            reward = episode['rewards'][i]

            # Store the transition
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

            # Calculate log probability
            norm_state = self.normalize_state(state)
            state_tensor = torch.FloatTensor(norm_state)
            action_probs = self.policy(state_tensor)
            dist = Categorical(action_probs)
            action_tensor = torch.tensor(action)
            self.log_probs.append(dist.log_prob(action_tensor))

        # Update policy with this good episode
        loss = self.update_policy(episode['total_reward'])
        print(f"Replayed good episode with length {len(episode['states'])} and loss {loss:.6f}")
        return loss

    def save_model(self, path):
        """Save model to file in a more compatible way"""
        # Save only the state dictionaries in a simple format
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'target_policy_state_dict': self.target_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Store numpy arrays as lists to avoid pickle issues
            'state_mean': self.state_mean.tolist() if hasattr(self.state_mean, 'tolist') else self.state_mean,
            'state_std': self.state_std.tolist() if hasattr(self.state_std, 'tolist') else self.state_std,
            'best_score': self.best_score,
            'best_avg_score': self.best_avg_score,
            'conservatism': self.conservatism
        }, path)

    def load_model(self, path):
        """Load model from file with multiple format support"""
        if os.path.exists(path):
            try:
                # Explicitly set weights_only=False for PyTorch 2.6+
                checkpoint = torch.load(path, weights_only=False)

                # Check which format the model was saved in
                if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                    # New format
                    self.policy.load_state_dict(checkpoint['policy_state_dict'])
                    if 'target_policy_state_dict' in checkpoint:
                        self.target_policy.load_state_dict(checkpoint['target_policy_state_dict'])
                    else:
                        self.target_policy = copy.deepcopy(self.policy)

                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    # Load other parameters
                    for key in ['state_mean', 'state_std', 'best_score', 'best_avg_score', 'conservatism']:
                        if key in checkpoint:
                            setattr(self, key, checkpoint[key])

                elif isinstance(checkpoint, dict) and 'policy' in checkpoint:
                    # Old format
                    self.policy.load_state_dict(checkpoint['policy'])
                    if 'target_policy' in checkpoint:
                        self.target_policy.load_state_dict(checkpoint['target_policy'])
                    else:
                        self.target_policy = copy.deepcopy(self.policy)

                    # Load other parameters
                    for key in ['state_mean', 'state_std', 'best_score', 'best_avg_score', 'conservatism', 'optimizer']:
                        if key in checkpoint:
                            setattr(self, key, checkpoint[key])

                # If it's just a state dict directly
                else:
                    self.policy.load_state_dict(checkpoint)
                    self.target_policy = copy.deepcopy(self.policy)

                print(f"Model loaded successfully from {path}")
                return True

            except Exception as e:
                print(f"Error loading model: {e}")
                # Try a simpler approach for older PyTorch versions
        return False


def shaped_reward(state, next_state, done, steps):
    """Simplified reward shaping that focuses on the pole angle"""
    x, x_dot, theta, theta_dot = next_state[0]

    # Base reward
    reward = 1.0

    # Only apply shaping after some initial steps
    if steps > 20:
        # Simple quadratic penalty on angle (the most important factor)
        angle_penalty = min(0.1, 5.0 * theta ** 2)

        # Very small position penalty (only for extreme positions)
        position_penalty = 0.0
        if abs(x) > 2.0:  # Only when far from center
            position_penalty = min(0.05, 0.02 * (abs(x) - 2.0))

        # Apply penalties
        reward -= (angle_penalty + position_penalty)

    # Extra penalty for falling
    if done and steps < MAX_STEPS - 1:
        reward -= 0.5  # Less harsh penalty

    return reward


def train():
    """Train the REINFORCE agent with safeguards against forgetting"""
    env = gym.make('CartPole-v1', render_mode=RENDER_MODE)
    env._max_episode_steps = 1500
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    agent = REINFORCEAgent(state_size, action_size)

    # Training tracking
    all_scores = []
    avg_scores = []
    losses = []

    # Success counter for termination condition
    success_counter = 0

    # Plateau detection
    plateau_length = 0
    last_best_avg = 0

    # Paths for models
    best_model_path = "./save_model/pg_cartpole_best.pt"
    solved_model_path = "./save_model/pg_cartpole_solved_1000.pt"
    checkpoint_model_path = "./save_model/pg_cartpole_checkpoint.pt"

    print("Starting training...")

    # Try to load existing model if available
    if os.path.exists(solved_model_path):
        print("Loading previously solved model...")
        agent.load_model(solved_model_path)
    elif os.path.exists(best_model_path):
        print("Loading best model...")
        agent.load_model(best_model_path)

    diversity_range = 0.05  # Start small

    for episode in range(EPISODES):
        # Gradually increase diversity range over training
        if episode % 100 == 0 and episode > 0:
            diversity_range = min(0.3, diversity_range + 0.02)

        # Custom environment reset with variable initialization
        state, _ = env.reset()

        # Add random perturbation to initial state
        if episode > 100:  # Start using diverse states after initial learning
            noise = np.random.uniform(-diversity_range, diversity_range, (1, state_size))
            state = state + noise
            state = np.clip(state, -0.5, 0.5)  # Keep within reasonable bounds

        state = np.reshape(state, [1, state_size])

        done = False
        score = 0

        # Episode tracking
        states_for_update = []
        actions_for_update = []
        rewards_for_update = []
        log_probs_for_update = []

        for step in range(MAX_STEPS):
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])

            # Calculate shaped reward
            reward = shaped_reward(state, next_state, done, step)

            # Store transition
            agent.store_transition(state, action, reward)

            # Update state and score
            state = next_state
            score += 1

            if done:
                break

        # Update policy after episode
        loss = agent.update_policy(score)
        losses.append(loss)

        # Track scores
        all_scores.append(score)
        avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
        avg_scores.append(avg_score)

        # Update plateau counter
        if score > agent.best_score:
            agent.best_score = score
            agent.plateau_counter = 0
        else:
            agent.plateau_counter += 1

        # Track average score plateaus for learning rate adjustments
        if avg_score > last_best_avg + 5:
            last_best_avg = avg_score
            plateau_length = 0
        else:
            plateau_length += 1

        # Print progress with more detail when doing well
        if score > 900 or episode % 10 == 0:
            print(f"Episode: {episode}, Score: {score}, Avg(100): {avg_score:.2f}, Best: {agent.best_score}, " +
                  f"Conservatism: {agent.conservatism:.2f}, LR: {agent.optimizer.param_groups[0]['lr']:.6f}")

        # Save best model based on average score
        if avg_score > agent.best_avg_score and len(all_scores) >= 100:
            agent.best_avg_score = avg_score
            print(f"New best average score: {avg_score:.2f}")
            agent.save_model(best_model_path)

        # Regular checkpoints
        if episode % 50 == 0:
            agent.save_model(checkpoint_model_path)

        # Handle plateaus with special techniques
        if agent.plateau_counter > 200 and episode % 10 == 0:
            print(f"Long plateau detected ({agent.plateau_counter} episodes). Applying recovery...")

            # Replay good episodes to reinforce successful behavior
            if agent.good_episodes:
                for _ in range(min(3, len(agent.good_episodes))):
                    agent.replay_good_episode()

            # Partially reset plateau counter to prevent excessive interventions
            agent.plateau_counter = max(100, agent.plateau_counter - 100)

        # Check for solving condition (1500+ score)
        if score >= 1500:
            success_counter += 1
            print(f"Success {success_counter}/150: Score {score}")

            # Save immediately on any 1500 score
            agent.save_model(solved_model_path)

            if success_counter >= 150:
                print(f"Solved! 1500+ scores achieved after {episode} episodes.")
                break
        else:
            # Decay counter, but not to zero to reward progress
            success_counter = max(0, success_counter - 0.25)

        # Create plots periodically
        if episode % 100 == 0:
            # === First figure with: Score trends, Recent scores, Loss ===
            plt.figure(figsize=(18, 5))

            # 1. Score and average score
            plt.subplot(1, 3, 1)
            plt.plot(all_scores, 'b-', alpha=0.3, label='Score')
            plt.plot(avg_scores, 'r-', label='Avg Score')
            plt.axhline(y=500, color='g', linestyle='--')
            plt.axhline(y=1000, color='g', linestyle='-')
            plt.title(f"Training Progress (Avg: {avg_score:.2f})")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.legend()

            # 2. Recent scores
            plt.subplot(1, 3, 2)
            recent_scores = all_scores[-100:] if len(all_scores) > 100 else all_scores
            plt.plot(recent_scores, label='Recent Scores')
            plt.title("Recent 100 Scores")
            plt.xlabel("Episode")
            plt.ylabel("Score")

            # 3. Losses
            plt.subplot(1, 3, 3)
            plt.plot(losses, 'purple', label='Policy Loss')
            plt.title("Policy Loss Over Time")
            plt.xlabel("Episode")
            plt.ylabel("Loss")

            plt.tight_layout()
            plt.savefig(f"./results/progress_metrics_A_episode_{episode}.png")
            plt.close()

            # === Second figure with: Learning rate, Conservatism, Reward distribution ===
            plt.figure(figsize=(18, 5))

            # 4. Learning rate
            lr_history = [agent.optimizer.param_groups[0]['lr']] * len(all_scores)
            plt.subplot(1, 3, 1)
            plt.plot(lr_history, 'orange')
            plt.title("Learning Rate")
            plt.xlabel("Episode")
            plt.ylabel("LR")

            # 5. Conservatism
            conservatism_history = [agent.conservatism] * len(all_scores)
            plt.subplot(1, 3, 2)
            plt.plot(conservatism_history, 'teal')
            plt.title("Conservatism Over Time")
            plt.xlabel("Episode")
            plt.ylabel("Conservatism")

            # 6. Histogram of scores
            plt.subplot(1, 3, 3)
            plt.hist(all_scores[-100:], bins=20, color='skyblue', edgecolor='black')
            plt.title("Reward Distribution (Recent 100)")
            plt.xlabel("Score")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.savefig(f"./results/progress_metrics_B_episode_{episode}.png")
            plt.close()

    return agent


def test(model_path=None):
    """Test the trained agent"""
    env = gym.make('CartPole-v1', render_mode="human")
    env._max_episode_steps = 10000
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    agent = REINFORCEAgent(state_size, action_size)


    # Try to load the best available model
    if model_path and os.path.exists(model_path):
        success = agent.load_model(model_path)
    else:
        success = False
        for path in [
            "./save_model/pg_cartpole_solved_1000.pt",
            "./save_model/pg_cartpole_best.pt",
            "./save_model/pg_cartpole_checkpoint.pt"
        ]:
            if os.path.exists(path):
                success = agent.load_model(path)
                if success:
                    print(f"Loaded model from {path}")
                    break

    if not success:
        print("No model found! Please train first.")
        return

    # Test for multiple episodes
    test_scores = []
    for episode in range(10):
        state, _ = env.reset()

        # Add small perturbation to test generalization
        noise = np.random.normal(0, 0.02, (1, state_size))
        state = state + noise

        state = np.reshape(state, [1, state_size])

        done = False
        score = 0

        for step in range(10000):
            env.render()

            # Select action (deterministic for testing)
            action = agent.select_action(state, training=False)

            # Take action
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated
            next_state = np.reshape(next_state, [1, state_size])

            state = next_state
            score += 1

            if done:
                break

        test_scores.append(score)
        print(f"Test Episode {episode + 1}: Score = {score}")

    avg_test_score = np.mean(test_scores)
    print(f"Average Test Score: {avg_test_score:.2f}")
    env.close()


if __name__ == "__main__":
    # Train agent
    agent = train()

    # Test agent
    test()