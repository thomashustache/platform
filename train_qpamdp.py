import os
os.environ['SDL_VIDEODRIVER']='dummy'
import sys
sys.path.append(os.getcwd() + '/')
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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
    env = PlatformEnv()
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
        max_optim_steps=1e5,
        max_plateau_episodes=7000,
        trainer_name=TRAINER_NAME,
    )

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
        max_optim_steps=1e5,
        max_plateau_episodes=3000,
        agent_id=1,
        trainer_name=TRAINER_NAME,
    )

    trainer = QPAMDPTrainer(
        env=env,
        experience_name=EXPERIENCE_NAME,
        logdir=LOGDIR,
        psearch_agent=a2c_agent,
        qlearn_agent=dqn_agent,
        max_qpamdp_steps=10,
        k_param=-1,
    )
    trainer.train()
