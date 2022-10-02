import os
os.environ['SDL_VIDEODRIVER']='dummy'
import sys
sys.path.append(os.getcwd() + '/')
from agents.DQNAgent import DQNAgent
from agents.A2CAgent import A2CAgent
from trainers.QPAMDPTrainer import QPAMDPTrainer
from utils.config import (
    EXPERIENCE_NAME,
    INIT_STDS,
    MIN_STDS,
    N_ACTIONS,
    SAVED_MODEL_DIR,
    OPTIMIZE_STDS,
    LOGDIR,
    DEVICE,
    INIT_ACTION_PARAMS,
    BATCH_SIZE_QLEARN,
    BATCH_SIZE_PSEARCH,
    LR_QLEARN,
    REPLAY_BUFFER_SIZE,
    LR_PSEARCH,
)

from gym_platform.envs.platform_env import PlatformEnv

if __name__ == "__main__":

    TRAINER_NAME = "QPAMDP"

    # 1. define Env
    env = PlatformEnv()

    # 2. Define Q-Learn Agent
    dqn_agent = DQNAgent(
        init_action_params=INIT_ACTION_PARAMS,
        nb_actions=N_ACTIONS,
        device=DEVICE,
        logdir=LOGDIR,
        saved_models_dir=SAVED_MODEL_DIR,
        memory_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE_QLEARN,
        lr=LR_QLEARN,
        agent_id=0,
        max_optim_steps=5e5,
        max_plateau_steps=7500,
    )

    # 3. Define P-search Agent
    a2c_agent = A2CAgent(
        init_action_params=INIT_ACTION_PARAMS,
        saved_models_dir=SAVED_MODEL_DIR,
        device=DEVICE,
        lr=LR_PSEARCH,
        logdir=LOGDIR,
        batch_size=BATCH_SIZE_PSEARCH,
        nb_actions=N_ACTIONS,
        optimize_stds=OPTIMIZE_STDS,
        init_stds=INIT_STDS,
        min_stds=MIN_STDS,
        max_optim_steps=5e5,
        max_plateau_steps=3000,
        agent_id=1,
    )

    # 4. define Q-PAMDP(k) trainer. k parameter is set to -1 which stands for the QPAMDP(inf) version
    trainer = QPAMDPTrainer(
        env=env,
        config_script='utils/config.py',
        experience_name=EXPERIENCE_NAME,
        logdir=LOGDIR,
        psearch_agent=a2c_agent,
        qlearn_agent=dqn_agent,
        max_qpamdp_steps=10,
        k_param=-1,
    )
    
    # 5. train
    trainer.train()
