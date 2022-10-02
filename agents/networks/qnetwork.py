import torch.nn as nn
import torch

from .xavier_init import xavier_init

class QNetwork(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 64):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, n_actions),
        )
    
        self.apply(xavier_init)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())

