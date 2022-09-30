import numpy as np
import torch
import torch.nn as nn

from api.transition import Transition
from api.action import Action, deconvert_act
from .BaseAgent import BaseAgent
from .agent_utils.memory import MovingAverageMemory, ReplayBuffer
from .networks.qnetwork import QNetwork
from gym_platform.envs.platform_env import ACTION_LOOKUP, PlatformEnv
from .agent_utils.torch_converter import torch_converter
from typing import Sequence
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.tensorboard import SummaryWriter

class DQNAgent(BaseAgent):
    def __init__(self,
                 memory_size: int,
                 lr: float,
                 epsilon: float = 0.5,
                 eps_decay: float = 0.995,
                 min_epsilon: float= 0.1,
                 update_freq: int = 20,
                 *args,
                 **kwargs,

                 ):
        """DQN Agent class

        Args:
            memory_size (int): Size of the replay Buffer
            epsilon (float, optional): Initial epsilon for eps-greedy policy. Defaults to 0.5.
            eps_decay (float, optional): decay rateof epsilon. Defaults to 0.999.
            min_epsilon (float, optional): minimum epsilon value to ensure exploration. Defaults to 0.1.
            update_freq (int, optional): frequency to update the TargetNetwork with the MainNetwork. Defaults to 20.
        """
        super(DQNAgent, self).__init__(algo_name='DQNAgent', *args, **kwargs)

        # main attributes
        self.epsilon = epsilon  # for eps-greedy policy
        self.init_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.eps_decay = eps_decay
        self.lr = lr
        self.update_freq = update_freq  # for Target network

        # Memory
        self.replay_buffer = ReplayBuffer(max_memory=memory_size, batch_size=self.batch_size, device=self.device)
        self.loss_memory = MovingAverageMemory(100)
        self.loss_writer_step = 0
        # self.batch_size = batch_size

        # Models
        self.main_qnet = QNetwork(obs_size=9, n_actions=self.nb_actions, hidden_size=128).to(self.device)
        self.target_qnet = QNetwork(obs_size=9, n_actions=self.nb_actions, hidden_size=128).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.main_qnet.parameters(), lr=lr)
        
    # def compute_action_weights(self, actions: torch.Tensor) -> torch.Tensor:
        
    #     actions_array = actions.detach().numpy()
    #     unique_actions = np.unique(actions_array)
    #     class_weights = compute_class_weight(class_weight='balanced',
    #                                 classes=unique_actions,
    #                                 y=actions_array)
        
    #     print(class_weights)
        
    #     return torch.Tensor(class_weights).to(self.device)

    def act(self, state: np.ndarray) -> int:
        """Choose an action with an eps-greedy policy

        Args:
            s (np.ndarray): current state

        Returns:
            int: action id
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.nb_actions)
        else:
            return self.learned_act(state)

    def learned_act(self, state: np.ndarray) -> int:
        """Select action accoring to the current policy

        Args:
            state (np.ndarray): current state

        Returns:
            (int): action id
        """

        torch_state = torch_converter(state).to(self.device)
        with torch.no_grad():
            a = np.argmax(self.main_qnet(torch_state).cpu().detach().numpy())

        return int(a)

    def update_target_graph(self) -> None:
        """In Deep Q-Learning, we maintain 2 networks: the Main_Qnetwork and the Target Network.
        We need to periodically update the Target network with the weights of the main network.
        """

        self.target_qnet.load_state_dict(self.main_qnet.state_dict())

    def update_epsilon(self) -> None:
        """Decrease epsilon value
        """
        self.epsilon = np.maximum(self.epsilon * self.eps_decay, self.min_epsilon)

    def run_one_episode(self, env: PlatformEnv, action_param_policy: BaseAgent = None) -> Sequence[Transition]:
        """Run one episode in the environment

        Args:
            env (PlatformEnv): the Platform environment

        Returns:
            (Sequence[Transition]): the collected experience of this episode
        """

        collected_xp = []
        d = False
        state, _ = env.reset()
        k = 0
        step_counter = 0
        episode_reward = 0
        while not d:
            action_id = self.act(state)

            # save action taken for tensorboard
            for act_id, a_name in ACTION_LOOKUP.items():
                if action_id == act_id:
                    self.action_mem[a_name].add(1)
                else:
                    self.action_mem[a_name].add(0)

            # select action parameters
            if action_param_policy is not None:
                action_params = action_param_policy.learned_act(state)
            else:
                action_params = self.init_action_params

            action = deconvert_act(Action(action_id, action_params[action_id]))
            (next_state, step), r, d, _, _ = env.step(action)
            step_counter += step
            episode_reward += r
            # discounted_rewards += self.discount ** k * r
            t = [state, action_id, r, next_state, d]
            collected_xp.append(Transition(*t))
            state = next_state
            k += 1

        # tensorboard logs
        self.rewardmem.add(episode_reward)
        self.stepmem.add(step_counter)
        self.nb_episodes_played += 1

        # self.check_need_saving()
        return collected_xp

    @torch.no_grad()
    def td_targets(self, next_states, rewards, dones) -> torch.Tensor:
        next_state_Q = self.main_qnet(next_states)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.target_qnet(next_states)[np.arange(0, self.batch_size), best_action]
        td_targets = (rewards + (1 - dones.float()) * self.discount * next_Q).float()

        return td_targets

    def single_optimization_step(self) -> None:
        """performs a single optimization step
        """

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        ###### OLD VERSION
        # # predict expected return of current state using main network
        # q_values = self.main_qnet(states)
        # pred_return, _ = torch.max(q_values, axis=1)

        # # get target return using target network
        # with torch.no_grad():
        #     q_values_next = self.target_qnet(next_states).detach()
        # max_q_values_next, _ = torch.max(q_values_next, axis=1)

        # # if done, we set Q(done) to 0
        # d = dones.squeeze(-1)
        # max_q_values_next[d == 1] = 0.0

        # td_target = rewards + self.discount * max_q_values_next

        pred_return = self.main_qnet(states)[np.arange(0, self.batch_size), actions]  # Q_online(s,a)
        td_target = self.td_targets(next_states=next_states, rewards=rewards, dones=dones)

        loss = nn.MSELoss()(pred_return, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        # loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()

    def init_training(self):
        self._init_training()
        self.replay_buffer.memory.clear()
        self.loss_memory.memory.clear()
        self.main_qnet.train()
        self.update_target_graph()
        self.epsilon = self.init_epsilon


    def train(self, env: PlatformEnv, action_param_agent: BaseAgent = None, verbose: bool = True) -> None:
        """Learning loop. Off-Policy update.

        Args:
            env (PlatformEnv): environment
            action_param_agent (BaseAgent, optional): The policy used to select the action parameters. Defaults to None.
        """

        self.init_training()
        has_converged = self.check_convergence()

        while self.current_optim_step < self.max_optim_steps and not has_converged:

            # collect experience
            xp = self.run_one_episode(env, action_param_policy=action_param_agent)
            self.replay_buffer.remember(xp)

            # when we have enough xp in the buffer, start training.
            if len(self.replay_buffer) >= self.batch_size:

                current_loss = self.single_optimization_step()
                self.loss_memory.add(current_loss)

                # Main Tensorboard logs
                if action_param_agent is not None:
                    action_param_agent.a2c_model.update_tensorboard(writer_step=self.writer_step)
                self.update_tensorboard_logs()

                # Specific tensorboard logs
                if self.current_optim_step % self.write_frequency == 0:
                    self.loss_writer.add_scalar(self.algo_name + '/mse_loss', self.loss_memory(), self.loss_writer_step)
                    self.loss_writer_step += 1

                # Update Target Network every n iterations
                if self.current_optim_step % self.update_freq == 0:
                    self.update_target_graph()
                    self.update_epsilon()

                # Verbose
                if self.current_optim_step % self.verbose_freq == 0 and verbose:
                    self.verbose(e=self.current_optim_step)

                # check convergence
                self.current_optim_step += 1
                has_converged = self.check_convergence()

    def verbose(self, e: int) -> None:
        """Some printings

        Args:
            e (int): current step
        """
        pass
        # print("Epoch {:03d}/{:03d} | Sum of discounted rewards : {:2.2f} | MSE loss : {:1.4f} | proba random action : {:1.2f}"
            #   .format(e, self.max_optim_steps, self.discounted_rewards[-1], self.losses[-1], self.epsilon))

    def save(self, model_name='best_model.pth') -> None:
        """Save torch model
        """
        print('saving DQN model...')
        torch.save(self.main_qnet.state_dict(), self.saved_models_dir + self.algo_name + '/' + model_name)

    def load(self, model_name='best_model.pth') -> None:
        """Load torch model
        """
        print('loading model...')
        self.main_qnet.load_state_dict(torch.load(self.saved_models_dir + self.algo_name + '/' + model_name))


