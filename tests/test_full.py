import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from policy_gradient import Policy, ValueFunction, PolicyGradientTrainer

def plot_rewards(rewards, window=100):
    """Plot rewards with moving average."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5, label='Raw rewards')
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("Starting comprehensive testing of Policy Gradient implementation...")
    
    # Hyperparameters (matching original notebook)
    alpha = 1e-2  # Policy learning rate
    beta = 1e-3   # Value function learning rate
    num_iterations = 300
    num_trajectories = 5
    gamma = 0.99
    
    # Create environment
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.low.size
    act_size = env.action_space.n
    
    # Initialize networks
    policy = Policy(obs_size, act_size, learning_rate=alpha)
    value_function = ValueFunction(obs_size, learning_rate=beta)
    
    # Create trainer
    trainer = PolicyGradientTrainer(
        policy=policy,
        value_function=value_function,
        env=env,
        gamma=gamma,
        use_wandb=False  # Disable wandb for testing
    )
    
    print("\nStarting training...")
    print(f"Training for {num_iterations} iterations with {num_trajectories} trajectories per iteration")
    
    # Record rewards for plotting
    training_rewards = []
    
    # Train the agent
    trainer.train(num_iterations=num_iterations, num_trajectories=num_trajectories)
    
    print("\nTraining completed!")
    
    # Evaluate final policy
    print("\nEvaluating final policy...")
    mean_reward = trainer.evaluate(num_episodes=100)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.1f}")
    
    # Success criteria (based on OpenAI Gym's definition for "solving" CartPole)
    is_solved = mean_reward >= 195.0
    print(f"\nEnvironment solved: {is_solved}")
    if is_solved:
        print("✓ Policy successfully learned to balance the pole!")
    else:
        print("✗ Policy did not fully solve the environment")
    
    env.close()

if __name__ == "__main__":
    main() 