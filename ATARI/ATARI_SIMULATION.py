import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
import skimage
from graphs import graph
import gym
import DQN
import CNN
##HYPERPARAMETERS
learning_rate = 0.00001
EPISODES = 5000
INPUTSIZE = (84,84)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)
   
def getFrame(x):
    x = x[25:210,0:160]
    state = skimage.color.rgb2gray(x)
    state = skimage.transform.resize(state, INPUTSIZE)
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    state = state.astype('uint8')
    return state

def makeState(state):
    return np.stack((state[0],state[1],state[2],state[3]), axis=0)

def train(game):
    env = gym.make(game)
    y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    target_y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    loss_fn = nn.HuberLoss()
    optimizer = optim.Adam(y.parameters(), lr = learning_rate)
    agent = DQN.DQN()
    state = deque(maxlen = 4)
    print(y)
    answer = input("Use a pre-trained model y/n? ")
    if answer == "y":
        agent.loadModel(y,'pixel_atari_weights.pth')
        agent.loadModel(target_y,'pixel_atari_weights.pth')
    frames_seen = 0
    rewards = []
    avgrewards = []
    loss = []
    for episode in range(1,EPISODES+500000000000):
        obs = env.reset()
        cumureward = 0
        lives = 5
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        while True:
            action = agent.getPrediction(makeState(state)/255,y)
            obs, reward, done, info = env.step(action)
            if info["ale.lives"] < lives:
                done = True
                lives -= 1
            env.render()
            cache = state.copy()
            state.append(getFrame(obs))
            agent.update_replay_memory((makeState(cache), action, reward, makeState(state), done))
            if len(agent.replay_memory) > 50000 and frames_seen % 4 == 0:
                loss.append(agent.train(y, target_y, loss_fn, optimizer))
                if frames_seen % 10000 == 0:
                    target_y.load_state_dict(y.state_dict())
                    print("Target net updated.")
            frames_seen+=1
            cumureward += reward
            if frames_seen % 10000 == 0:
                agent.saveModel(y,'pixel_atari_weights.pth')
            if done and lives == 0:
                break
        rewards.append(cumureward)
        avgrewards.append(np.sum(np.array(rewards))/episode)
        print("Score:", cumureward, " Episode:", episode, " frames_seen:", frames_seen , " Epsilon:", agent.EPSILON)
        if episode % 1000 == 0:
            graph(rewards, avgrewards, loss, "fetajuusto/DQN-FLAPPY-PIXEL")

if __name__ == "__main__":
    game = 'BreakoutDeterministic-v4'
    train(game)