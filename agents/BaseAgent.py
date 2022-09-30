import numpy as np
import os

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List
from agents.agent_utils.memory import MovingAverageMemory
from api.action import Action
from api.observation import Observation
from gym_platform.envs.platform_env import ACTION_LOOKUP, Constants, PlatformEnv
import torch
from torch.utils.tensorboard import SummaryWriter


from utils.config import INIT_ACTION_PARAMS, SAVED_MODEL_DIR


class BaseAgent(object):
    """ Agent Class
    """

    def __init__(
        self,
        max_optim_steps: float,
        max_plateau_episodes: int,
        agent_id: int,
        algo_name: str,
        device: torch.device,
        trainer_name: str = "",
        discount: float = 0.99,
        batch_size: int = 64,
        saved_models_dir: str = SAVED_MODEL_DIR,
        logdir: str = 'logs',
        init_action_params: Sequence[float] = INIT_ACTION_PARAMS,
        nb_actions: int = len(ACTION_LOOKUP),
        verbose_freq: int = 10,
        tb_writer: SummaryWriter = None,
    ):
        """Base Agent. This class will be overwritten to define several different agents.

        Args:
            action_params (Sequence[float]): the initial values of the continuous action parameters.
            saved_models_dir (str): folder where the models' weights has to be saved
            max_optim_steps (int): maximum number of optimization step.
            max_plateau_episodes (int): maximum number of episodes to consider the algo has converged.
            discount (float, optional): Discount factor. Defaults to 0.99.
            batch_size (int, optional): batch size. Defaults to 64.
            nb_action (int, optional): number of discrete actions. Defaults to len(ACTION_LOOKUP).
            verbose_freq (int, optional): frequency of printing scores on the screen. Defaults to 10.
            reward_writer (SummaryWriter, optional): tensorboard writer to trace the evolution of the reward. Defaults to None.
        """

        # main attributes
        self.trainer_name = trainer_name
        self.device = device
        self.agent_id = agent_id
        self.algo_name = algo_name
        self.batch_size = batch_size
        self.discount = discount
        self.max_optim_steps = max_optim_steps
        self.current_optim_step = 0
        self.nb_actions = nb_actions
        self.init_action_params = init_action_params
        self.max_plateau_episodes = max_plateau_episodes
        self.convergence_counter = 0
        

        # secondary attributes
        self.saved_models_dir = saved_models_dir
        self.best_avg_reward = 0
        self.verbose_freq = verbose_freq

        # tensorboard attributes
        self.logdir = logdir
        self.nb_episodes_played = 0
        self.writer_step = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.write_frequency = 10
        self.tb_writer = tb_writer
        self.action_mem = {
            a_name: MovingAverageMemory(250) for a_name in ACTION_LOOKUP.values()
        }
        self.rewardmem = MovingAverageMemory(max_memory=250)
        self.stepmem = MovingAverageMemory(max_memory=250)
        self.loss_writer = SummaryWriter(log_dir=self.logdir + '/' + self.algo_name)
        self._build_save_directory()

    @abstractmethod
    def act(self, state: np.ndarray) -> Action:
        """Choose an action

        Args:
            s (np.ndarray): state
            train (bool, optional): Train mode?. Defaults to True.

        Raises:
            NotImplementedError: _description_

        Returns:
            Action: Action to take
        """

        pass

    @abstractmethod
    def train(self, env: PlatformEnv, verbose: bool = True) -> None:
        """Train method.

        Args:
            env (PlatformEnv): The env on which to train the agent
            verbose (bool): If we want to display results on screen

        """
        pass

    @abstractmethod
    def learned_act(self, state: np.ndarray) -> Action:
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""
        pass

    @abstractmethod
    def save(self, model_name: str) -> None:
        """ This function returns basic stats if applicable: the
        loss and/or the model"""
        pass

    @abstractmethod
    def load(self, model_name: str) -> None:
        """ This function allows to restore a model"""
        pass

    def _build_save_directory(self) -> None:
        os.makedirs(self.saved_models_dir + self.algo_name, exist_ok=True)

    def _init_training(self) -> None:
        """Training initialization
        """
        self.best_avg_reward = 0
        self.current_optim_step = 0
        self.nb_episodes_played = 0
        self.convergence_counter = 0
        self.rewardmem.memory.clear()

    def update_tensorboard_logs(self):
        if (
            self.tb_writer is not None
            and self.current_optim_step % self.write_frequency == 0
        ):
            self.tb_writer.add_scalar(
                "Env/mean_episode_reward", self.rewardmem(), self.writer_step
            )
            self.tb_writer.add_scalar(
                "Env/mean_episode_steps", self.stepmem(), self.writer_step
            )
            self.tb_writer.add_scalars(
                "Env/actions_percentage",
                {k: a() for k, a in self.action_mem.items()},
                self.writer_step,
            )
            self.tb_writer.add_scalar(
                "Env/which_agent_training", self.agent_id, self.writer_step
            )
            self.writer_step += 1

    def check_need_to_save(self) -> None:
        """Check if we have to save model weights
        """
        pass
        # avg_reward = self.rewardmem()
        # if avg_reward > self.best_avg_reward:
        #     self.save(model_name='best_model.pth')
        #     self.highest_reward = self.best_avg_reward

    def check_convergence(self, delta_reward: float = 0.005) -> bool:
        """Check if the average reward per episode does not increase anymore
        """

        if self.convergence_counter > self.max_plateau_episodes:
            self.convergence_counter = 0
            self.best_avg_reward = 0  # TODO: check this
            print(
                f"Stopping training because reward has not increased during the last {self.max_plateau_episodes} episodes"
            )
            return True

        current_avg_reward = self.rewardmem()
        if current_avg_reward < self.best_avg_reward + delta_reward:
            # if the avg_reward doesnt improve from best_avg_reward + delta, we increment the convergence_counter
            self.convergence_counter += 1
        else:
            print(
                f"best average reward has improved from {self.best_avg_reward} to {current_avg_reward}"
            )
            self.convergence_counter = 0
            self.best_avg_reward = current_avg_reward
            self.save(model_name="best_model.pth")

        return False
