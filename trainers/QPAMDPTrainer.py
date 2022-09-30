from genericpath import exists
from agents.DQNAgent import DQNAgent
from agents.A2CAgent import A2CAgent
from gym_platform.envs.platform_env import PlatformEnv
from torch.utils.tensorboard import SummaryWriter
from agents.agent_utils.torch_converter import torch_converter
from api.action import Action, deconvert_act
import os
import glob

from utils.frames_to_vid import load_images_to_video
from utils.video_to_gif import make_gif_from_video
from utils.clean_dir import clean_dir



class QPAMDPTrainer(object):

    def __init__(self,
                 env: PlatformEnv,
                 logdir: str,
                 experience_name:str,
                 psearch_agent: A2CAgent,
                 qlearn_agent: DQNAgent,
                 k_param: int = 1,
                 max_qpamdp_steps: int = 10) -> None:

        # agents
        self.env = env
        self.psearch_agent = psearch_agent
        self.qlearn_agent = qlearn_agent


        # optim attributes
        self.max_steps = max_qpamdp_steps
        self.k_param = k_param

        # tensorboard attributes
        self.experience_name = experience_name
        self.name = 'QPAMDP'
        self.psearch_agent.trainer_name = self.name
        self.qlearn_agent.trainer_name = self.name
        self.writer = SummaryWriter(log_dir=logdir + f'/{self.name}')
        self.tot_writter_steps = 0
        self.global_optim_step = 0
        self.tot_episodes_played = 0
        self.eval_frequency = 1


        self.set_common_tensorboard_writers()

    def set_common_tensorboard_writers(self):

        self.qlearn_agent.tb_writer = self.writer
        self.psearch_agent.tb_writer = self.writer

    def record_video_policy(self, first_call: bool = False) -> None:

        print('generating video...')
        self.psearch_agent.load('best_model.pth')
        self.psearch_agent.a2c_model.eval()
        self.qlearn_agent.load('best_model.pth')
        self.qlearn_agent.main_qnet.eval()
        env = PlatformEnv()

        video_path = f'results/videos/{self.name}/{self.experience_name}/'
        gif_folder = f'results/gifs/{self.name}/{self.experience_name}/'
        frames_path = f'results/frames/{self.name}/{self.experience_name}/optimstep_{self.global_optim_step}/'
        
        if first_call:
            os.makedirs(frames_path, exist_ok=True)
            os.makedirs(video_path, exist_ok=True)
            os.makedirs(gif_folder, exist_ok=True)
            clean_dir(frames_path)
            clean_dir(video_path)
            clean_dir(gif_folder)

        tot_r = 0
        d = False

        state, _ = env.reset()

        while not d:

            action_id = self.qlearn_agent.learned_act(state)
            params = self.psearch_agent.learned_act(state)
            action = deconvert_act(Action(action_id, params[action_id]))
            (state, _), r, d, _, _ = env.step(action)
            tot_r += r

        tot_r = str(int(tot_r * 1000) / 1000)

        # Save all frames and create video + Gif
        video_name = video_path + f'optimstep_{self.global_optim_step}_reward-' + tot_r + '.mp4'
        env.save_render_states(dir=frames_path, prefix='')
        load_images_to_video(imgs_path=frames_path, video_name=video_name)

        # Gif
        print('generating gif from video...')
        gif_name = f'optimstep_{self.global_optim_step}_reward-' + tot_r + '.gif'
        make_gif_from_video(video_path=video_name,
                            gif_folder=gif_folder,
                            gif_name=gif_name,
                            jpgs_frame_folder='results/jpgs/',
                            skip_rate=2)


    def train(self):
        """Reproduce the training procedure described in the paper: https://arxiv.org/pdf/1509.01644.pdf

        Inputs:

            Inital weights Theta0 (Action Parameters policy) and W0 (Action policy)
            Policy Search method (to update Theta)
            Q-Learn algorithm (to update W)

        Algorithm:

            w <- Q-Learn(oo) with fixed theta
            Repeat:
                theta <- P-Search(k) with fixed w
                w <- Q-Learn(oo) with fixed theta
            until theta converges

        """

        # 1. w <- Q-Learn(oo) with fixed theta
        self.psearch_agent.a2c_model.eval()
        self.qlearn_agent.train(env=self.env, action_param_agent=self.psearch_agent, verbose=False)

        # 2. Repeat until theta converges
        self.record_video_policy(first_call=True)
        self.global_optim_step += 1
        while self.global_optim_step < self.max_steps:

            ## update all tb writer steps
            self.psearch_agent.writer_step = self.qlearn_agent.writer_step
            # self.psearch_agent.rewardmem = self.qlearn_agent.rewardmem
            # self.psearch_agent.stepmem = self.qlearn_agent.stepmem
            # self.psearch_agent.action_mem = self.qlearn_agent.action_mem

            print('Training Psearch algorithm...')
            self.qlearn_agent.load(model_name='best_model.pth')
            self.qlearn_agent.main_qnet.eval()
            self.psearch_agent.a2c_model.train()
            if self.k_param == 1:
                self.psearch_agent.train_one_step(env=self.env, action_id_agent=self.qlearn_agent)
            else:
                self.psearch_agent.train(env=self.env, action_id_agent=self.qlearn_agent, verbose=False)

            ## update all tb writer steps
            self.qlearn_agent.writer_step = self.psearch_agent.writer_step
            # self.qlearn_agent.rewardmem = self.psearch_agent.rewardmem
            # self.qlearn_agent.stepmem = self.psearch_agent.stepmem
            # self.qlearn_agent.action_mem = self.psearch_agent.action_mem

            print('Training Qlearn algorithm...')
            self.psearch_agent.load(model_name='best_model.pth')
            self.psearch_agent.a2c_model.eval()
            self.qlearn_agent.train(env=self.env, action_param_agent=self.psearch_agent)

            #record video
            self.record_video_policy()
            self.global_optim_step += 1






