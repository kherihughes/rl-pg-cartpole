import gymnasium as gym
import wandb
from policy_gradient import Policy, ValueFunction, PolicyGradientTrainer

def main():
    # Initialize wandb (optional)
    wandb.init(
        project="policy-gradient-example",
        config={
            "env_name": "CartPole-v1",
            "policy_lr": 0.01,
            "value_lr": 0.001,
            "gamma": 0.99,
            "num_iterations": 300,
            "num_trajectories": 5
        }
    )

    # Create environment
    env = gym.make("CartPole-v1")
    
    # Initialize networks
    policy = Policy(
        obs_size=env.observation_space.low.size,
        act_size=env.action_space.n,
        learning_rate=0.01
    )
    
    value_function = ValueFunction(
        obs_size=env.observation_space.low.size,
        learning_rate=0.001
    )
    
    # Create trainer
    trainer = PolicyGradientTrainer(
        policy=policy,
        value_function=value_function,
        env=env,
        gamma=0.99
    )
    
    print("\nStarting training...")
    print("Training until the agent solves CartPole-v1 (mean reward >= 195.0)")
    
    # Train the agent
    trainer.train(num_iterations=300, num_trajectories=5)
    
    # Evaluate the trained policy
    mean_reward = trainer.evaluate(num_episodes=100)
    print(f"\nFinal evaluation over 100 episodes:")
    print(f"Mean Reward: {mean_reward:.1f}")
    
    # Check if environment is solved
    is_solved = mean_reward >= 195.0
    print(f"\nEnvironment solved: {is_solved}")
    if is_solved:
        print("✓ Policy successfully learned to balance the pole!")
        print(f"  Achieved reward of {mean_reward:.1f} (threshold is 195.0)")
    else:
        print("✗ Policy did not fully solve the environment")
    
    # Close environment and wandb
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main() 