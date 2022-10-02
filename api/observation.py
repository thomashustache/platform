from typing import NamedTuple, Tuple

import numpy as np

class BasicFeatures(NamedTuple):
    player_pos: float
    player_vel: float
    enemy_pos: float
    enemy_vel: float

class PlatformsFeatures(NamedTuple):
    wd1: float
    wd2: float
    gap: float
    pos: float
    diff: float

class Observation(NamedTuple):
    basic_features: BasicFeatures
    platforms_features: PlatformsFeatures
    step: int

def convert_obs(s: Tuple[np.ndarray, int]) -> Observation:
    state, steps = s[0], s[1]

    return Observation(basic_features=BasicFeatures(*state[:4]),
                       platforms_features=PlatformsFeatures(*state[4:]),
                       step=steps)

def deconvert_obs(s: Observation) -> Tuple[np.ndarray, int]:
    basic_features = [s.basic_features.player_pos,
                      s.basic_features.player_vel,
                      s.basic_features.enemy_pos,
                      s.basic_features.enemy_vel]

    plateforms_features = [s.platforms_features.wd1,
                           s.platforms_features.wd2,
                           s.platforms_features.gap,
                           s.platforms_features.pos,
                           s.platforms_features.diff]

    state = np.concatenate((basic_features, plateforms_features))
    return (state, s.step)


