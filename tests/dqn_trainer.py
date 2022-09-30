import gym
from gym_platform.envs.platform_env import PlatformEnv
from agents.BaseAgent import BaseAgent
from agents.DQNAgent import DQNAgent
from api.action import Action, deconvert_act, convert_act
# import cv2
import glob
import numpy as np




if __name__ == '__main__':
    
    env = gym.make('Platform-v0')
    agent = DQNAgent(epsilon=0.1, memory_size=100, batch_size=16, nb_actions=3)
    
    agent.train(env=env)
    