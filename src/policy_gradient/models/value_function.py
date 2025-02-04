import torch
import numpy as np

class ValueFunction:
    """
    Value function network for policy gradient with baseline.
    
    Args:
        obs_size (int): Size of the observation space
        learning_rate (float): Learning rate for the optimizer
        hidden_size (int, optional): Size of the hidden layer. Defaults to 128.
    """
    def __init__(self, obs_size, learning_rate, hidden_size=128):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.obs_size = obs_size

    def compute_values(self, states):
        """
        Compute value function for given states.
        
        Args:
            states (numpy.ndarray): Array of states with shape [batch_size, obs_size]
            
        Returns:
            numpy.ndarray: Value function estimates with shape [batch_size]
        """
        states = torch.FloatTensor(states)
        return self.model(states).cpu().detach().numpy()

    def train(self, states, targets):
        """
        Train the value function network.
        
        Args:
            states (numpy.ndarray): Array of states
            targets (numpy.ndarray): Array of target values
            
        Returns:
            float: Loss value
        """
        states = torch.FloatTensor(states)
        targets = torch.FloatTensor(targets)

        # Compute value predictions
        v_preds = self.model(states)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(v_preds.squeeze(), targets)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy() 