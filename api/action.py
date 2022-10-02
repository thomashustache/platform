from typing import NamedTuple, List
from gym_platform.envs.platform_env import ACTION_LOOKUP
import numpy as np

class Action(NamedTuple):

    id: int
    param: float

def convert_act(action: np.ndarray) -> Action:

    act_index = action[0]
    param = action[1][act_index][0]

    return Action(act_index, param)

def deconvert_act(action: Action) -> List:
    act_index = action.id
    param = action.param

    params = [[0]] * len(ACTION_LOOKUP)
    params[act_index][0] = param

    return [act_index, params]

