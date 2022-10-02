from agents.DQNAgent import DQNAgent
from agents.A2CAgent import A2CAgent
from gym_platform.envs.platform_env import PlatformEnv
from torch.utils.tensorboard import SummaryWriter
from api.action import Action, deconvert_act
import os

from utils.frames_to_vid import load_images_to_video
from utils.video_to_gif import make_gif_from_video
from utils.clean_dir import clean_dir
from utils.py_script_to_txt import save_config_file


class QPAMDPTrainer(object):

    def __init__(self,
                 env: PlatformEnv,
                 config_script: str,
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
        self.writer = SummaryWriter(log_dir=logdir + f'/{self.name}')
        self.tot_writter_steps = 0
        self.global_optim_step = 0
        
        # config
        self.config_script = config_script


        self.set_common_tensorboard_writers()

    def set_common_tensorboard_writers(self) -> None:

        self.qlearn_agent.tb_writer = self.writer
        self.psearch_agent.tb_writer = self.writer

    def record_video_policy(self, first_call: bool = False) -> None:

        print('generating video...')
        if not first_call:
            self.psearch_agent.load('best_model.pth')
        
        # pass all models in inference mode.
        self.psearch_agent.model.eval()
        self.qlearn_agent.load('best_model.pth')
        self.qlearn_agent.model.eval()
        env = PlatformEnv()

        # Make sure all the paths are well created to save the videos, gifs, frames etc
        video_path = f'results/videos/{self.name}/{self.experience_name}/'
        gif_folder = f'results/gifs/{self.name}/{self.experience_name}/'
        frames_path = f'results/frames/{self.name}/{self.experience_name}/optimstep_{self.global_optim_step}/'
        os.makedirs(frames_path, exist_ok=True)
        if first_call:
            os.makedirs(video_path, exist_ok=True)
            os.makedirs(gif_folder, exist_ok=True)
            clean_dir(frames_path)
            clean_dir(video_path)
            clean_dir(gif_folder)
            save_config_file(self.config_script, video_path + 'config.txt')
        
        
        # start recording an episode
        total_reward = 0
        d = False
        state, _ = env.reset()
        while not d:

            action_id = self.qlearn_agent.learned_act(state)
            params = self.psearch_agent.learned_act(state)
            action = deconvert_act(Action(action_id, params[action_id]))
            (state, _), r, d, _, _ = env.step(action)
            total_reward += r

        total_reward = str(int(total_reward * 100) / 100)

        # Save all frames and create video
        video_name = video_path + f'optimstep_{self.global_optim_step}_reward-' + total_reward + '.mp4'
        env.save_render_states(dir=frames_path, prefix='')
        load_images_to_video(imgs_path=frames_path, video_name=video_name)
        
        # rm frames once finished
        clean_dir(frames_path)
            

        # Gif from video
        print('generating gif from video...')
        gif_name = f'optimstep_{self.global_optim_step}_reward-' + total_reward + '.gif'
        make_gif_from_video(video_path=video_name,
                            gif_folder=gif_folder,
                            gif_name=gif_name,
                            jpgs_frame_folder='results/jpgs/',
                            skip_rate=5)


    def train(self) -> None:
        """Reproduce the Q-PAMDP(k) training procedure described in the paper: https://arxiv.org/pdf/1509.01644.pdf

        Inputs:

            Initial weights Theta0 (Action Parameters policy) and W0 (Action policy)
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
        self.psearch_agent.model.eval()
        self.qlearn_agent.train(env=self.env, action_param_agent=self.psearch_agent)

        # 2. Repeat until theta converges
        self.record_video_policy(first_call=True)
        self.global_optim_step += 1
        while self.global_optim_step < self.max_steps:

            ## update all tb writer steps
            self.psearch_agent.writer_step = self.qlearn_agent.writer_step
            print('Training Psearch algorithm...')
            self.qlearn_agent.load(model_name='best_model.pth')
            self.qlearn_agent.model.eval()
            self.psearch_agent.model.train()
            if self.k_param != -1:
                for _ in range(self.k_param):
                    self.psearch_agent.train_one_step(env=self.env, action_id_agent=self.qlearn_agent)
            else:
                self.psearch_agent.train(env=self.env, action_id_agent=self.qlearn_agent)

            
            self.qlearn_agent.writer_step = self.psearch_agent.writer_step
            print('Training Qlearn algorithm...')
            self.psearch_agent.load(model_name='best_model.pth')
            self.psearch_agent.model.eval()
            self.qlearn_agent.train(env=self.env, action_param_agent=self.psearch_agent)

            # record current policy video
            self.record_video_policy()
            self.global_optim_step += 1






