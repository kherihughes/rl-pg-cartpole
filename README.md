# Policy Gradient Implementation in PyTorch

A high-performance implementation of Policy Gradient with Value Function Baseline using PyTorch. This implementation consistently solves the CartPole-v1 environment, achieving a mean reward of ~500 (well above the solving threshold of 195).

## Features

- Policy Gradient implementation with Value Function Baseline
- Support for discrete action spaces
- Configurable hyperparameters and network architectures
- Integration with OpenAI Gymnasium environments
- Weights & Biases logging support for experiment tracking
- Advantage normalization for stable training
- Progress monitoring and evaluation tools

## Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/kherihughes/rl-pg-cartpole.git
cd rl-pg-cartpole
pip install -r requirements.txt
```

## Quick Start

```python
import gymnasium as gym
from policy_gradient import Policy, ValueFunction, PolicyGradientTrainer

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

# Train the agent
trainer.train(num_iterations=300, num_trajectories=5)

# Evaluate
mean_reward = trainer.evaluate(num_episodes=100)
print(f"Mean evaluation reward: {mean_reward}")
```

## Performance

On the CartPole-v1 environment, this implementation typically:
- Solves the environment (average reward > 195 over 100 episodes) within approximately 30-50 training iterations.
- Reaches near-maximum performance (average reward approaching 500) with continued training.
- Shows stable learning dynamics due to the value function baseline and advantage normalization.
- Works well with the provided default hyperparameters.

## Project Structure

```
policy-gradient-pytorch/
├── src/
│   └── policy_gradient/
│       ├── models/
│       │   ├── policy.py        # Policy network implementation
│       │   └── value_function.py # Value function network
│       ├── utils/
│       │   └── rewards.py       # Reward computation utilities
│       └── trainer.py           # Main training logic
├── examples/
│   └── cartpole.py             # Example training script
├── tests/
│   ├── test_basic.py           # Basic functionality tests
│   └── test_full.py            # Full training tests
├── requirements.txt
└── README.md
```

## Key Components

- **Policy Network**: Implements the policy π(a|s) as a neural network that outputs action probabilities
- **Value Function**: Estimates the state value V(s) to reduce variance in policy gradients
- **Trainer**: Handles the training loop, data collection, and policy updates
- **Utilities**: Includes functions for reward discounting and advantage normalization

## Hyperparameters

The default hyperparameters are tuned for the CartPole environment:
- Policy Learning Rate: 0.01
- Value Function Learning Rate: 0.001
- Discount Factor (γ): 0.99
- Training Iterations: 300
- Trajectories per Iteration: 5

## Contributing

Contributions are welcome! Some areas for potential improvements:
- Support for continuous action spaces
- Implementation of PPO and other policy gradient variants
- Multi-environment training support
- More extensive testing and benchmarking

## License

This project is licensed under the MIT License - see the LICENSE file for details. 