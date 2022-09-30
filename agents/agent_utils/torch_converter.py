import torch
import numpy as np


def torch_converter(x):
    
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    return torch.from_numpy(x).float()