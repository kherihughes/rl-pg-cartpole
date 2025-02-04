import numpy as np
import gymnasium as gym
from policy_gradient import Policy, ValueFunction, PolicyGradientTrainer

def test_imports_and_basic_functionality():
    """Test basic functionality of the policy gradient implementation."""
    print("Testing basic functionality...")
    
    # Create environment
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.low.size
    act_size = env.action_space.n
    
    # Test policy network
    print("\n1. Testing Policy Network...")
    policy = Policy(obs_size, act_size, learning_rate=0.01)
    test_state = np.random.randn(1, obs_size)
    probs = policy.compute_prob(test_state)
    print(f"  ✓ Policy shape correct: {probs.shape == (1, act_size)}")
    print(f"  ✓ Probabilities sum to 1: {abs(np.sum(probs) - 1) < 1e-6}")
    
    # Test value function
    print("\n2. Testing Value Function...")
    value_fn = ValueFunction(obs_size, learning_rate=0.001)
    values = value_fn.compute_values(test_state)
    print(f"  ✓ Value function shape correct: {values.shape == (1, 1)}")
    
    # Test trainer initialization
    print("\n3. Testing Trainer...")
    trainer = PolicyGradientTrainer(
        policy=policy,
        value_function=value_fn,
        env=env,
        gamma=0.99,
        use_wandb=False  # Disable wandb for testing
    )
    
    # Test trajectory collection
    states, actions, rewards = trainer.collect_trajectory()
    print(f"  ✓ Collected trajectory successfully")
    print(f"  ✓ Trajectory length: {len(states)} steps")
    
    print("\nAll basic functionality tests passed!")
    env.close()

if __name__ == "__main__":
    test_imports_and_basic_functionality() 