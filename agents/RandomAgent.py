from .BaseAgent import BaseAgent

import numpy as np
from typing import Tuple
from api.action import Action

from api.observation import Observation

class RandomAgent(BaseAgent):
    """ Agent Class
    Args:

        epsilon (float): parameter of the epsilon-greddy policy
        n_action (int): number of possible actions (an integer from 0 to n_action not included)
    """

    def __init__(self, epsilon: float = 0.1,
                 n_action: int = 3,
                 params_min: np.ndarray = np.array([0, 0, 0]),
                 params_max: np.ndarray = np.array([30, 720, 430])):
        super().__init__(epsilon, n_action, params_min, params_max)



    def learned_act(self, s: np.ndarray) -> Action:
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""

        act_index = np.random.randint(0, self.n_action, size=1)[0]
        param = np.random.uniform(self.params_min[act_index], self.params_max[act_index])

        return Action(id=act_index, param=param)


    def save(self):
        pass

    def load(self):
        pass