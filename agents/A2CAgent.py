import torch
import numpy as np

from agents.agent_utils.memory import MovingAverageMemory
from .BaseAgent import BaseAgent
from gym_platform.envs.platform_env import PlatformEnv, ACTION_LOOKUP
from api.action import deconvert_act, Action
from .networks.actor_critic import ActorCritic
from .agent_utils.torch_converter import torch_converter


class A2CAgent(BaseAgent):
    def __init__(self,
                 lr: float,
                 init_stds: torch.Tensor,
                 min_stds: torch.Tensor,
                 optimize_stds: bool = True,
                 *args,
                 **kwargs
                 ):
        """Episodic Advantage Actor-Critic

        Args:
            lr (float): learning rate
            init_stds (torch.Tensor): Inital values for Stds
            min_stds (torch.Tensor): _description_
            optimize_stds (bool, optional): _description_. Defaults to True.
        """
        super(A2CAgent, self).__init__(algo_name='A2CAgent', *args, **kwargs)

        # model
        self.optimize_stds = optimize_stds
        self.min_stds = min_stds.to(self.device)
        self.init_stds = init_stds.to(self.device)
        self.std_decay = 0.999 # useful when Stds are fixed. We manually decrease their values over training.
        init_mean_action_params = torch.Tensor(self.init_action_params).to(self.device)
        self.model = ActorCritic(act_size=self.nb_actions,
                                  logdir=self.logdir,
                                  hidden_size=128,
                                  optimize_stds=optimize_stds,
                                  min_stds=self.min_stds,
                                  init_means=init_mean_action_params,
                                  init_stds=self.init_stds.clone()).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

        # counters
        self.memory = [] # we will fill this with batch_size transitions
        self.step_counter = 0

        #tensorboard
        self.actorloss_mem = MovingAverageMemory(100)
        self.criticloss_mem = MovingAverageMemory(100)
        self.loss_writer_step = 0

    def learned_act(self, state: np.ndarray) -> np.ndarray:
        """Select action according to current policy

        Args:
            state (np.ndarray): state on which to predict the next action

        Returns:
            np.ndarray: continuous action parameters (The platform env will clamp them if they are out of range)
        """

        with torch.no_grad():
            torch_sate = torch_converter(state).to(self.device)
            dists, _ = self.model(torch_sate)
            action_params = dists.sample().detach().cpu().data.numpy()

            return action_params


    def act(self, state: np.ndarray) -> torch.Tensor:
        """Sample action from current policy's distribution

        Args:
            state (np.ndarray): state

        Returns:
            torch.Tensor: action parameters
        """

        torch_sate = torch_converter(state).to(self.device)
        dists, _ = self.model(torch_sate)
        action_params = dists.sample()

        return action_params

    def train_one_step(self, env: PlatformEnv, action_id_agent: BaseAgent) -> None:

        """Collect a batch of transitions to perform a single parameters update.
        On-Policy algorithm: The current optimization step is performed based on data collected with the current policy.
        """

        state = env.get_state()
        critic_loss = 0
        actor_loss = 0
        k = 0
        reward = 0

        for _ in range(self.batch_size):

            # Choose action id according to fixed policy from Q-Learning agent
            action_id = action_id_agent.learned_act(state)
            
            # Update action choices statistics for display
            for act_id, a_name in ACTION_LOOKUP.items():
                if action_id == act_id:
                    self.action_mem[a_name].add(1)
                else:
                    self.action_mem[a_name].add(0)

            # Compute action parameters from Actor's head and value from Critic's head
            torch_sate = torch_converter(state).to(self.device)
            dists, value_pred = self.model(torch_sate)
            action_params = dists.sample()

            # compute log_prob for update rule
            log_prob = dists.log_prob(action_params)[action_id]

            # Update of the environment
            params = action_params.clone().detach().cpu().data.numpy()
            action = deconvert_act(Action(action_id, param=params[action_id]))
            (next_state, step), reward, done, _, _ = env.step(action)
            torch_sate = torch_converter(next_state).to(self.device)
            _, next_value_pred = self.model(torch_sate)

            # Basic 1-step Advantage estimation: A(s, a, theta) = (r + gamma * V(s', theta)) - V(s, theta)
            # TODO: n-step return estimation, GAE
            advantage = reward + (1 - done) * self.discount * next_value_pred - value_pred
            critic_loss += advantage.pow(2)
            actor_loss -= log_prob * advantage.detach()

            state = next_state
            self.episode_reward += reward
            self.episode_step += step
            k += 1
            if done:
                state, _ = env.reset()
                self.rewardmem.add(self.episode_reward)
                self.stepmem.add(self.episode_step)
                k = 0
                self.episode_reward = 0
                self.episode_step = 0

        # parameters update
        self.optimizer.zero_grad()
        total_loss = (critic_loss + actor_loss) / self.batch_size
        total_loss.backward()
        self.optimizer.step()

        # update scalars
        self.criticloss_mem.add(critic_loss.item() / self.batch_size)
        self.actorloss_mem.add(actor_loss.item() / self.batch_size)
        
    def init_training(self):
        """Training initialization
        """
        self._init_training()
        self.model.train()
        self.model.init_stds = self.init_stds.clone() # when stsds are not optimized, we reset the stds to their initial values to ensure exploration

    def train(self, env: PlatformEnv, action_id_agent: BaseAgent) -> None:
        """Main training loop. On-policy update.

        Args:
            env (PlatformEnv): environment
            action_id_agent (BaseAgent): Qlearn Agent
        """

        self.init_training()
        while self.current_optim_step < self.max_optim_steps and not self.check_convergence():
            self.train_one_step(env=env, action_id_agent=action_id_agent)

            # Main Tensorboard logs
            self.update_tensorboard_logs()

            # Specific Logs
            if self.current_optim_step % self.write_frequency == 0:
                self.loss_writer.add_scalar(self.algo_name + f'/critic_loss', self.criticloss_mem(), self.current_optim_step)
                self.loss_writer.add_scalar(self.algo_name + f'/actor_loss', self.actorloss_mem(), self.current_optim_step)
                self.model.update_tensorboard(writer_step=self.writer_step)

            self.current_optim_step += 1
            
            # decrease Stds values with decay rate
            if not self.optimize_stds:
                self.model.init_stds = torch.amax(torch.stack([self.model.init_stds * self.std_decay, self.min_stds], 0), 0)






