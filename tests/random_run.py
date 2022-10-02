import gym
from gym_platform.envs.platform_env import PlatformEnv
from agents.BaseAgent import BaseAgent
from agents.RandomAgent import RandomAgent
from api.action import deconvert_act


def run(agent: BaseAgent, env: PlatformEnv, epoch: int, prefix: str = ''):
    # Number of won games
    score = 0
    loss = 0

    score_per_epoch = []


    for e in range(epoch):
        # At each epoch, we restart to a fresh game and get the initial state
        state, _ = env.reset()
        # state = convert_obs(s=(state, 0))
        # This assumes that the games will terminate
        game_over = False

        win = 0
        lose = 0

        while not game_over:
            # The agent performs an action
            action = agent.act(state)
            action = deconvert_act(action)

            # Apply an action to the environment, get the next state, the reward
            # and if the games end
            # prev_state = state
            (state, step), reward, game_over, _, info = env.step(action)
            # state = convert_obs(s=state)

            # Update the counters
            if reward > 0:
                win = win + reward
            if reward < 0:
                lose = lose -reward

        # Save as a mp4
        env.save_render_states(dir='runs/random_agent/', prefix='')

        # Update stats
        score += win-lose

        # ADD
        score_per_epoch.append(win-lose)
        # ADD

        print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win/lose count {}/{} ({})"
              .format(e, epoch, loss, win, lose, win-lose))

    return score_per_epoch

if __name__ == '__main__':
    env = gym.make('Platform-v0')
    agent = RandomAgent()

    run(agent=agent, env=env , epoch=1)

    # load_images_to_video(imgs_path='runs/random_agent/')