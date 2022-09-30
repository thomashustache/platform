import numpy as np
import torch
import torch.nn as nn
from agents.agent_utils.memory import MovingAverageMemory
from gym_platform.envs.platform_env import Constants, ACTION_LOOKUP
from utils.config import DEVICE
from torch.utils.tensorboard import SummaryWriter
from utils.config import INIT_ACTION_PARAMS, LOGDIR

# max_means = torch.from_numpy(Constants.PARAMETERS_MAX).to(DEVICE)
# max_stds = torch.Tensor([3, 3, 3]).to(DEVICE) # adjusted by looking at this : https://academo.org/demos/gaussian-distribution/

class ActorCritic(nn.Module):
    def __init__(self,
                 init_stds: torch.Tensor,
                 init_means: torch.Tensor,
                 logdir: str,
                 obs_size: int = 9,
                 act_size: int = 3,
                 hidden_size: int = 128,
                 optimize_stds: bool = False):
        """Actor-Critic model

        Args:
            init_stds (torch.Tensor): Stds Init values of each discrete action's parameters
            init_means (torch.Tensor): Means Init values of each discrete action's parameters
            logdir (str): log directory for tensorboard
            obs_size (int, optional): State size. Defaults to 9.
            act_size (int, optional): Nb of discrete actions. Defaults to 3.
            hidden_size (int, optional): Hidden layers' size. Defaults to 128.
            optimize_stds (bool, optional): Wether we optimize stds or fix them. Defaults to False.
        """
        super(ActorCritic, self).__init__()


        self.log_writer = SummaryWriter(logdir + '/A2CAgent')
        self.mu_mem = {}
        self.sigma_mem = {}
        self.writer_step = 0
        self.nb_calls = 0
        self.optimize_stds = optimize_stds
        self.init_means = init_means
        self.init_stds = init_stds
        self.calibration_means = torch.Tensor([1., 1., 1.]).to(DEVICE)
        self.calibration_stds = torch.Tensor([1., 1., 1.]).to(DEVICE)
        self._init_memories()

        self.base = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden_size, act_size),
            nn.ReLU(),
            # Initally was nn.Tanh() but it makes more sense to use Relu to have positive values for mu parameters
        )

        self.std = nn.Sequential(
            nn.Linear(hidden_size, act_size),
            nn.ReLU()
            # Initially was nn.Softplus(), 
        )
        self.value = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)


    def _init_memories(self) -> None:
        """Initialize buffers.
        """
        for _, a_name in ACTION_LOOKUP.items():
            self.mu_mem[a_name] = MovingAverageMemory(100)
            self.sigma_mem[a_name] = MovingAverageMemory(100)
        

    def _init_weights(self, module):
        """Custom weight Initialization to guide the agent at the very beginning so that he takes reasonable action parameters.

        Args:
            module (_type_): _description_
        """
        if isinstance(module, nn.Linear):
            # module.weight.data.fill_(0.03)
            module.weight.data.normal_(mean=0.1, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def _store_params(self, mus: torch.Tensor, stds: torch.Tensor):
        """Store the output Means and Stds in their respective buffers.

        Args:
            mus (Tensor): Action param Means that are output by the model.
            stds (Tensor): Action param Stds that are output by the model.
        """
        for i, a_name in ACTION_LOOKUP.items():
            self.mu_mem[a_name].add(mus[i].clone().item())
            self.sigma_mem[a_name].add(stds[i].clone().item())

    def update_tensorboard(self, writer_step: int):
        """Update tensorboard logs.

        Args:
            writer_step (int): current step
        """

        self.log_writer.add_scalars('Params/Means_ActParam/', {k:v() for k, v in self.mu_mem.items()}, writer_step)
        self.log_writer.add_scalars('Params/Sigmas_ActParam/', {k:v() for k, v in self.sigma_mem.items()}, writer_step)

    def forward(self, x: torch.Tensor) -> torch.distributions:
        base_out = self.base(x)

        mus = self.mu(base_out) * self.calibration_means
        stds = self.std(base_out) * self.calibration_stds if self.optimize_stds else self.init_stds
        
        # if first_call of the forward method, we rescale the outputs of the model so that they are equal to the desired means and stds.
        if self.nb_calls == 0:
            for ind, y in enumerate(self.init_means):
                self.calibration_means[ind] = y / mus[ind]
            mus = self.mu(base_out) * self.calibration_means
            
            if self.optimize_stds:
                for ind, y in enumerate(self.init_stds):
                    self.calibration_stds[ind] = y / stds[ind]
                stds = self.std(base_out) * self.calibration_stds
            self.nb_calls = 1
            
        self._store_params(mus=mus, stds=stds)
        return torch.distributions.Normal(mus, stds), self.value(base_out)




