import numpy as np

def compute_discounted_rewards(rewards, gamma):
    """
    Compute discounted rewards for a sequence of rewards.
    
    Args:
        rewards (list): List of rewards
        gamma (float): Discount factor
        
    Returns:
        numpy.ndarray: Array of discounted rewards
    """
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_sum = 0
    
    for i in reversed(range(len(rewards))):
        discounted_rewards[i] = running_sum * gamma + rewards[i]
        running_sum = discounted_rewards[i]
        
    return discounted_rewards

def normalize_rewards(rewards):
    """
    Normalize rewards to have zero mean and unit variance.
    
    Args:
        rewards (numpy.ndarray): Array of rewards
        
    Returns:
        numpy.ndarray: Normalized rewards
    """
    rewards = np.array(rewards)
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8) 