import torch
import numpy as np

class Policy:
    """
    Policy network for discrete action spaces.
    
    Args:
        obs_size (int): Size of the observation space
        act_size (int): Size of the action space
        learning_rate (float): Learning rate for the optimizer
        hidden_size (int, optional): Size of the hidden layer. Defaults to 128.
    """
    def __init__(self, obs_size, act_size, learning_rate, hidden_size=128):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, act_size)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.obs_size = obs_size
        self.act_size = act_size

    def compute_prob(self, states):
        """
        Compute probability distribution over actions given states.
        
        Args:
            states (numpy.ndarray): Array of states with shape [batch_size, obs_size]
            
        Returns:
            numpy.ndarray: Probability distribution over actions with shape [batch_size, act_size]
        """
        states = torch.FloatTensor(states)
        prob = torch.nn.functional.softmax(self.model(states), dim=-1)
        return prob.cpu().detach().numpy()

    def _to_one_hot(self, y, num_classes):
        """Convert an integer vector to one-hot representation."""
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def train(self, states, actions, advantages):
        """
        Train the policy network using policy gradient.
        
        Args:
            states (numpy.ndarray): Array of states
            actions (numpy.ndarray): Array of actions taken
            advantages (numpy.ndarray): Array of advantage estimates
            
        Returns:
            float: Loss value
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        advantages = torch.FloatTensor(advantages)

        # Compute action probabilities
        logits = self.model(states)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        # Compute selected action probabilities
        action_onehot = self._to_one_hot(actions, self.act_size)
        prob_selected = torch.sum(prob * action_onehot, axis=-1)

        # Add small constant for numerical stability
        prob_selected = prob_selected + 1e-8

        # Compute policy gradient loss
        loss = -torch.mean(advantages * torch.log(prob_selected))

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy() 