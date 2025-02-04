import numpy as np
import wandb
from .utils.rewards import compute_discounted_rewards, normalize_rewards

class PolicyGradientTrainer:
    """
    Trainer class for Policy Gradient with Value Function Baseline.
    
    Args:
        policy: Policy network instance
        value_function: Value function network instance
        env: OpenAI Gym environment
        gamma (float): Discount factor
        use_wandb (bool, optional): Whether to use Weights & Biases logging. Defaults to True.
    """
    def __init__(self, policy, value_function, env, gamma, use_wandb=True):
        self.policy = policy
        self.value_function = value_function
        self.env = env
        self.gamma = gamma
        self.use_wandb = use_wandb

    def collect_trajectory(self):
        """Collect a single trajectory using current policy."""
        states, actions, rewards = [], [], []
        
        # Handle both old and new gym API
        state, _ = self.env.reset()
        done = False

        while not done:
            # Get action probabilities and sample action
            probs = self.policy.compute_prob(np.array([state]))
            action = np.random.choice(self.env.action_space.n, p=probs.flatten())

            # Take action in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        return np.array(states), np.array(actions), np.array(rewards)

    def train(self, num_iterations, num_trajectories):
        """
        Train the policy and value function.
        
        Args:
            num_iterations (int): Number of training iterations
            num_trajectories (int): Number of trajectories to collect per iteration
        """
        for iteration in range(num_iterations):
            all_states, all_actions, all_values = [], [], []
            episode_rewards = []

            # Collect trajectories
            for _ in range(num_trajectories):
                states, actions, rewards = self.collect_trajectory()
                discounted_rewards = compute_discounted_rewards(rewards, self.gamma)
                
                all_states.extend(states)
                all_actions.extend(actions)
                all_values.extend(discounted_rewards)
                episode_rewards.append(np.sum(rewards))

            # Convert to numpy arrays
            all_states = np.array(all_states)
            all_actions = np.array(all_actions)
            all_values = np.array(all_values)

            # Update value function
            value_loss = self.value_function.train(all_states, all_values)

            # Compute advantages
            baselines = self.value_function.compute_values(all_states).squeeze()
            advantages = all_values - baselines

            # Normalize advantages
            advantages = normalize_rewards(advantages)

            # Update policy
            policy_loss = self.policy.train(all_states, all_actions, advantages)

            # Progress reporting
            mean_episode_reward = np.mean(episode_rewards)
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}, Average Reward: {mean_episode_reward:.1f}")

            # Logging
            if self.use_wandb:
                wandb.log({
                    'iteration': iteration,
                    'mean_episode_reward': mean_episode_reward,
                    'policy_loss': policy_loss,
                    'value_loss': value_loss
                })

    def evaluate(self, num_episodes=100):
        """
        Evaluate the current policy.
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            float: Mean episode reward
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                probs = self.policy.compute_prob(np.array([state]))
                action = np.random.choice(self.env.action_space.n, p=probs.flatten())
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            
            if self.use_wandb:
                wandb.log({'eval_reward': episode_reward})
        
        mean_reward = np.mean(total_rewards)
        if self.use_wandb:
            wandb.run.summary['mean_eval_reward'] = mean_reward
            
        return mean_reward 