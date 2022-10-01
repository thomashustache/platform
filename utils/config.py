from gym_platform.envs.platform_env import Constants, ACTION_LOOKUP
import numpy as np
import torch

N_ACTIONS = len(ACTION_LOOKUP)
EXPERIENCE_NAME = "optimized_std"
SAVED_MODEL_DIR = "results/saved_models/" + EXPERIENCE_NAME + "/"
LOGDIR = 'results/logs/' + EXPERIENCE_NAME
DEVICE = torch.device("cuda:0")

OPTIMIZE_STDS = True
INIT_ACTION_PARAMS = [3.0, 150.0, 200.0] # setting low values on purpose
# INIT_ACTION_PARAMS = [np.random.uniform(Constants.PARAMETERS_MIN[i], Constants.PARAMETERS_MAX[i]) for i in range(N_ACTIONS)]
# INIT_STDS = torch.Tensor([1, 1, 1])
INIT_STDS = torch.Tensor(0.20 * Constants.PARAMETERS_MAX) # useful to help exploration 
MIN_STDS = torch.Tensor(0.025 * Constants.PARAMETERS_MAX) # not useful when OPTIMIZE_STDS = True. The model has to learn this itself.

# MIN_STDS = torch.Tensor([0.75, 0.75, 0.75])
BATCH_SIZE_QLEARN = 128
BATCH_SIZE_PSEARCH = 128
REPLAY_BUFFER_SIZE = 50000
LR_QLEARN = 2e-4
LR_PSEARCH = 2e-4
MAX_OPTIM_STEPS = 1e6
