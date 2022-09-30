import numpy as np
import torch

from .BaseAgent import BaseAgent
from api.action import Action

class RandomAgent(BaseAgent):

    def __init__(self, min_parameters: np.ndarray,
                 max_parameters: np.ndarray,
                 *args,
                 **kwargs,
                 ):
        super().__init__(saved_models_dir='',
                         max_plateau_steps=0,
                         max_optim_steps=0,
                         agent_id=2,
                         algo_name='random',
                         init_action_params=[],
                         device=torch.device('cpu')
                         *args,
                 **kwargs,)
        
        self.params_min = min_parameters
        self.params_max = max_parameters

    def act(self, s: np.ndarray):
        pass
    
    def learned_act(self, s: np.ndarray) -> Action:

        act_index = np.random.randint(0, self.nb_actions, size=1)[0]
        param = np.random.uniform(self.params_min[act_index], self.params_max[act_index])

        return Action(id=act_index, param=param)

    def save(self):
        pass

    def load(self):
        pass