import torch.nn as nn

def xavier_init(module):
    """Xavier Initialization: https://paperswithcode.com/method/xavier-initialization 
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()