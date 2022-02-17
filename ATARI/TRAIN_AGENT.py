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
import DQN
import CNN
from process_state import clip_reward, makeState, getFrame
import hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)



def train(game):
    env = gym.make(game)
    y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    target_y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    loss_fn = nn.HuberLoss()
    optimizer = optim.Adam(y.parameters(), lr = hyperparameters.learning_rate)
    agent = DQN.DQN(hyperparameters.REPLAY_MEMORY_SIZE, hyperparameters.BATCH_SIZE, hyperparameters.GAMMA, hyperparameters.EPSILON, hyperparameters.EPSILON_MIN, hyperparameters.EPSILON_DECAY)
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
    wandb.init(project="BREAKOUT_DQN", entity="neuroori") 
    for episode in range(1,hyperparameters.EPISODES+500000000000):
        obs = env.reset()
        cumureward = 0
        lives = 3 ## 5 for breakout, 3 for spaceinvaders, 0 for pong, 3 for robotank :D
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        while True:
            action = agent.getPrediction(makeState(state)/255,y)
            ##Repeat same action four times for flappybird/doom otherwise set it to one.
            #for repeat in range(4):
            obs, reward, done, info = env.step(action)
            #    if done or reward == 1:
            #        break
            ##uncomment for atari
            if info["ale.lives"] < lives:
                done = True
                lives -= 1
            env.render()
            cache = state.copy()
            state.append(getFrame(obs))
            agent.update_replay_memory((makeState(cache), action, clip_reward(reward), makeState(state), done))
            ##Train the agent
            if len(agent.replay_memory) >= hyperparameters.START_TRAINING_AT_STEP and frames_seen % hyperparameters.TRAINING_FREQUENCY == 0:
                loss.append(agent.train(y, target_y, loss_fn, optimizer))
            ##Update target network  
            if len(agent.replay_memory) >= hyperparameters.START_TRAINING_AT_STEP and frames_seen % hyperparameters.TARGET_NET_UPDATE_FREQUENCY == 0:
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
        wandb.log({"Reward per episode":cumureward, "Avg reward":(np.sum(np.array(rewards))/episode), "Loss":loss})
        if len(loss)>0:
            loss = []

if __name__ == "__main__":
    #game = 'BreakoutDeterministic-v4'
    #game = "SpaceInvadersDeterministic-v4"
    #game = "PongDeterministic-v4"
    game = "RobotankDeterministic-v4"
    #game = 'FlappyBird-v0'
    train(game)