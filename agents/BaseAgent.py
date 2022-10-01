import numpy as np
import os
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Sequence
from agents.agent_utils.memory import MovingAverageMemory
from api.action import Action
from gym_platform.envs.platform_env import ACTION_LOOKUP, PlatformEnv
import torch
from torch.utils.tensorboard import SummaryWriter



class BaseAgent(ABC):

    def __init__(
        self,
        max_optim_steps: float,
        max_plateau_steps: int,
        agent_id: int,
        algo_name: str,
        init_action_params: Sequence[float],
        device: torch.device,
        saved_models_dir: str,
        discount: float = 0.99,
        batch_size: int = 64,
        logdir: str = 'results/logs',
        nb_actions: int = len(ACTION_LOOKUP),
        tb_writer: SummaryWriter = None,
    ):
        """Base Agent. This class will be overwritten to define several different agents.

        Args:
            max_optim_steps (float): maximum number of optimization step allowed for the learning algorithm
            max_plateau_steps (int): maximum number of optimization steps that has to be reached without improving the policy to consider the convergence.
            agent_id (int): id of the agent (useful for tensorboard displaying)
            algo_name (str): name of the agent
            init_action_params (Sequence[float]): Inital values of the action parameters.
            device (torch.device): GPU or CPU
            saved_models_dir (str): directory where the weights of the models are going to be saved 
            discount (float, optional): disctount factor. Defaults to 0.99.
            batch_size (int, optional): Defaults to 64.
            logdir (str, optional): tensorboard log directory. Defaults to 'results/logs'.
            nb_actions (int, optional): . Defaults to len(ACTION_LOOKUP).
            tb_writer (SummaryWriter, optional): tensorboard SummaryWriter. Defaults to None.
        """

        # main attributes
        self.model : nn.Module = None
        self.device = device
        self.agent_id = agent_id
        self.algo_name = algo_name
        self.batch_size = batch_size
        self.discount = discount
        self.max_optim_steps = max_optim_steps
        self.current_optim_step = 0
        self.nb_actions = nb_actions
        self.init_action_params = init_action_params
        self.max_plateau_steps = max_plateau_steps
        self.convergence_counter = 0
        
        # secondary attributes
        self.saved_models_dir = saved_models_dir
        self.best_avg_reward = 0

        # tensorboard attributes
        self.logdir = logdir
        self.writer_step = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.write_frequency = 10
        self.tb_writer = tb_writer
        self.action_mem = {
            a_name: MovingAverageMemory(250) for a_name in ACTION_LOOKUP.values()
        } # For each action, keep trace of the nb of times it was taken over the optimization steps
        self.rewardmem = MovingAverageMemory(max_memory=250) # trace the average reward over optimization steps
        self.stepmem = MovingAverageMemory(max_memory=250) # trace the average nb of steps per episode over optimization steps
        self.loss_writer = SummaryWriter(log_dir=self.logdir + '/' + self.algo_name)
        self._build_save_directory()

    @abstractmethod
    def act(self, state: np.ndarray) -> Action:
        """Choose an action to an eps-greedy policy.

        Args:
            s (np.ndarray): state

        Raises:
            NotImplementedError: _description_

        Returns:
            Action: Action to take
        """

        pass

    @abstractmethod
    def train(self, env: PlatformEnv) -> None:
        """Train method.

        Args:
            env (PlatformEnv): The env on which to train the agent

        """
        pass

    @abstractmethod
    def learned_act(self, state: np.ndarray) -> Action:
        """ Act according to current agent's policy. From a given state s
        it proposes an action a"""
        pass


    def _build_save_directory(self) -> None:
        """Build directories and sub directories to save our models later.
        """
        os.makedirs(self.saved_models_dir + self.algo_name, exist_ok=True)

    def _init_training(self) -> None:
        """Training initialization
        """
        self.best_avg_reward = 0
        self.current_optim_step = 0
        self.convergence_counter = 0
        self.rewardmem.memory.clear()

    def update_tensorboard_logs(self) -> None:
        """When an optimization step is done, we update all of our tensorboard logs.
        """
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
            

    def check_convergence(self, delta_reward: float = 0.005) -> bool:
        """Check if the average reward per episode does not increase anymore so that we can declare the algorithm has converged.

        Args:
            delta_reward (float, optional): The margin needed to reach to consider the average rewards has increased. Defaults to 0.005.

        Returns:
            bool: Wether the convergence has been reached or not.
        """
        
        if len(self.rewardmem) < self.rewardmem.max_memory:
            # To avoid too high variance, we wait for the buffer to be full before updating the best_avg_reward variable
            return False

        if self.convergence_counter > self.max_plateau_steps:
            self.convergence_counter = 0
            self.best_avg_reward = 0 
            print(
                f"Stopping training because reward has not increased during the last {self.max_plateau_steps} episodes"
            )
            return True

        current_avg_reward = self.rewardmem()
        if current_avg_reward < self.best_avg_reward + delta_reward:
            # if the avg_reward doesnt improve from best_avg_reward + delta, we increment the convergence_counter
            self.convergence_counter += 1
        else:
            print(
                f"Best average reward per episode has improved from {self.best_avg_reward} to {current_avg_reward}. Saving model."
            )
            self.convergence_counter = 0
            self.best_avg_reward = current_avg_reward
            self.save(model_name="best_model.pth")

        return False
    
    def save(self, model_name: str = 'best_model.pth') -> None:
        """Save torch model.

        Args:
            model_name (str, optional): _description_. Defaults to 'best_model.pth'.
        """
        assert(type(model_name) is str), f'model_name parameter must be a str not {type(model_name)}'
        print(f'Saving {self.algo_name} model...')
        torch.save(self.model.state_dict(), self.saved_models_dir + self.algo_name + '/' + model_name)

    def load(self, model_name: str ='best_model.pth') -> None:
        """Load torch model
        """
        assert(type(model_name) is str), f'model_name parameter must be a str not {type(model_name)}'
        print(f'Loading {self.algo_name} model...')
        self.model.load_state_dict(torch.load(self.saved_models_dir + self.algo_name + '/' + model_name))
