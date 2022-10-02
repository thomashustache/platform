from gym_platform.envs.platform_env import PlatformEnv
from agents.BaseAgent import BaseAgent
from agents.DQNAgent import DQNAgent

import numpy as np
import torch
import gym


from utils.config import INIT_ACTION_PARAMS, DEVICE, LOGDIR, SAVED_MODEL_DIR

if __name__ == '__main__':
    
    env = gym.make('Platform-v0')
    device = torch.device('cuda:0')
    dqn_agent = DQNAgent(
        init_action_params=INIT_ACTION_PARAMS,
        nb_actions=3,
        device=DEVICE,
        logdir=LOGDIR,
        saved_models_dir=SAVED_MODEL_DIR,
        memory_size=10000,
        batch_size=128,
        lr=2e-4,
        agent_id=0,
        max_optim_steps=5e5,
        max_plateau_steps=7500,
    )
    
    dqn_agent.train(env=env)
    