from typing import NamedTuple, Tuple
from .observation import Observation, convert_obs, deconvert_obs
from .action import Action, convert_act, deconvert_act
import numpy as np

class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# def convert_transition(t: Tuple[np.ndarray, np.ndarray, float, np.ndarray, int, bool]) -> Transition:

#     s = t[0]
#     a = t[1]
#     r = t[2]
#     next_s = t[3]
#     step = t[4]
#     game_over = t[5]

#     act = convert_act(a)
#     state = convert_obs((s, step))
#     next_sate = convert_obs((next_s, step + 1))

#     return Transition(state=state, action=act, reward=r, next_state=next_sate, game_over=game_over)

# def deconvert_transition(t: Transition) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, int, bool]:

#     (state, step)= deconvert_obs(s=t.state)
#     next_state = deconvert_obs(s=t.next_state)
#     action = deconvert_act(action=t.action)

#     return state, action, t.reward, next_state, step, t.game_over

