# Reinforcement Learning Platform Environment.

This repository was forked from the original: https://github.com/cycraig/gym-platform/.

## 1. Installation.

First, you will need to clone the entire repository: ```git clone https://github.com/thomashustache/platform/```.

Then, you need to install the conda virtual environment and all of its dependancies from the .yml file:
```conda env create --name cycraig --file=env.yml```

## 2. Algorithm Overview

- The algorithm I chose to implement is called **QPAMDP** and is fully described in the paper [Masson et al. 2016](https://arxiv.org/pdf/1509.01644.pdf).
- QPAMDP has two components:
  1. a Q-learning algorithm that will optimize the discrete action choices. I chose a Deep Q-learning with Experience Replay, whose implementation can be found in the script ```agents/DQNAgent.py```
  2. a Policy Search algorithm that will optimize the continuous parameter choices of each action. I chose the Advantage Actor-Critic algorithm, whose implementation ca be found in the script ```agents/A2CAgent.py```
- More precisely, A2C tries to optimize a gaussian distribution for each of the discrete actions' parameter's space, that is to say learning the optimal ***means*** and ***standard deviation*** of each gaussian distribution.
- I considered the action's parameters as independant variables, that is why I restricted the A2C model to only learn means and variances, and not to learn covariances.
  
## 3. Launch Training

- First, you can adjust some hyperparameters at your taste in the ```utils/config.py``` file. For example adjusting the batch size, the learning rates, etc.
- Note that there is a parameter called 'OPTIMIZED_STDS'. If true, the A2C will have to learn both Means AND Standard deviations. If False, the A2C will only learn the means parameters, and the standard deviations will be fixed and manually decreased over training steps.
- Then, a training of the QPAMDP algorithm can be launched with the ``` python train_qpamdp.py``` command.

## 4. Monitoring training with Tensorboard

## 5. Tracks of improvements
