from collections import deque
from typing import Sequence, Tuple
from api.transition import Transition
import numpy as np
import torch
import random


class ReplayBuffer(object):
    def __init__(self, max_memory: int, device: torch.device, batch_size: int):
        """ReplayBuffer for the DQN algorithm.

        Args:
            max_memory (int, optional): Max Buffer Size. Defaults to 100.
        """
        self.max_memory = max_memory
        # self.memory = deque(maxlen=max_memory)
        self.memory = []
        self.batch_size = batch_size
        self.device = device

    def remember(self, t: Sequence[Transition]) -> None:
        """store a transition in the buffer

        Args:
            t (Sequence[Transition]): a sequence of (s, r, a, s') in the API format 'Transition'. 
            
            Note: A lot of repositories are using the 'deque' list format as the buffer but to generate more iid data it is better not to use this format.
        """

        self.memory += t
        if len(self) > self.max_memory:
            diff = len(self) - self.max_memory
            np.random.shuffle(self.memory)
            self.memory = self.memory[:-diff]
        # self.memory.append(t)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random sample a batch from the memory"""

        batch_data = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array([t.state for t in batch_data])).to(self.device)
        rewards = torch.from_numpy(np.array([t.reward for t in batch_data])).to(self.device)
        next_states = torch.from_numpy(np.array([t.next_state for t in batch_data])).to(
            self.device
        )
        actions = torch.from_numpy(np.array([t.action for t in batch_data])).to(self.device)
        dones = torch.from_numpy(np.array([t.done for t in batch_data])).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """overload length operator"""
        return len(self.memory)


class MovingAverageMemory(object):
    def __init__(self, max_memory: int) -> None:
        """Keep trace of the moving average of a specific variable (reward, loss, parameter value, etc) that will be displayed on tensorboard.

        Args:
            max_memory (int): maximum size length of the buffer. The average will be computed based on this number.
            The higher it is, the smoother the plot.
        """
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)

    def add(self, value: float) -> None:
        """Add a new value to the buffer

        Args:
            value (float): variable you want to keep trace of .
        """
        self.memory.append(value)

    def __call__(self) -> float:
        """Overload call operator.

        Returns:
            float: the average value of the buffer.
        """
        return np.mean(self.memory)
    
    def __len__(self) -> float:
        """overload length operator"""
        return len(self.memory)
