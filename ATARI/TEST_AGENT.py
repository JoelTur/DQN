import torch
from torch import nn
from torch._C import device
from torch import optim
import numpy as np
from collections import deque
from wandb import wandb
from graphs import graph
import gym
import gym_ple
import vizdoomgym
import DQN
import CNN
import hyperparameters
from process_state import getFrame, clip_reward, makeState
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)

def test(game):
    env = gym.make(game)
    y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    agent = DQN.DQN(hyperparameters.REPLAY_MEMORY_SIZE, hyperparameters.BATCH_SIZE, hyperparameters.GAMMA, 0, 0, hyperparameters.EPSILON_DECAY)
    agent.loadModel(y,game +'.pth')
    state = deque(maxlen = 4)
    print(y)
    lives = 0
    score = 0
    obs = env.reset()
    state.append(getFrame(obs))
    state.append(getFrame(obs))
    state.append(getFrame(obs))
    state.append(getFrame(obs))
    rewards = []
    avgrewards = []
    cumureward = 0
    games = 1
    while True:
        action = agent.getPrediction(makeState(state)/255,y)
        for i in range(4):
            obs, reward, done, info = env.step(action)
            if done or reward == 1:
               break
        state.append(getFrame(obs))
        score += reward
        cumureward += reward
        env.render()
        #time.sleep(1/30)
        #if info["ale.lives"] < lives:
        #    done = True
        #    lives -= 1
        if done and lives == 0:
            lives = 0
            obs = env.reset()
            #obs, reward, done, info = env.step(1)
            state.append(getFrame(obs))
            state.append(getFrame(obs))
            state.append(getFrame(obs))
            state.append(getFrame(obs))
            print("Score: ", score, " Game: " , games)
            score = 0
            rewards.append(cumureward)
            avgrewards.append(np.sum(np.array(rewards))/games)
            cumureward = 0
            games += 1
        if games == 100:
            break
    print(np.sum(np.array(rewards))/games)

if __name__ == "__main__":
    #game = 'BreakoutDeterministic-v4'
    #game = "SpaceInvadersDeterministic-v4"
    #game = "PongDeterministic-v4"
    #game = "RobotankDeterministic-v4"
    game = "VizdoomDefendCenter-v0"
    #game = "VizdoomDeathmatch-v0"
    #game = 'FlappyBird-v0'
    test(game)