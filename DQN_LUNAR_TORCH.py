import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
import gym

##HYPERPARAMETERS
learning_rate = 0.0001
REPLAY_MEMORY_SIZE=100000
BATCH_SIZE = 64
GAMMA = 0.99
EPISODES = 5000
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = (EPSILON-EPSILON_MIN)/30000
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)

class NeuralNetwork(nn.Module):
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        super(NeuralNetwork, self).__init__()

        self.Q = nn.Sequential(nn.Linear(self.obsSpaceSize, 64),
        nn.ReLU(),
        nn.Linear(64, self.actionSpaceSize))

    def forward(self, x):
        x = x.to(device)
        Q_values = self.Q(x)
        return Q_values 


class DQN(nn.Module):

    def __init__(self):
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.ddqn = True

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, agent, target, loss_fn, optimizer):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        X = []
        Y = []
        optimizer.zero_grad()
        y = [agent(torch.from_numpy(np.array(transition[0]))) for transition in batch]
        target_y = [target(torch.from_numpy(np.array(transition[3]))) for transition in batch]
        y_next = [agent(torch.from_numpy(np.array(transition[3]))) for transition in batch]
        for i,(state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            elif self.ddqn is False:
                y[i][action] = reward + GAMMA*torch.max(target_y[i])
            else:
                y[i][action] = reward + GAMMA*target_y[i][torch.argmax(y_next[i])]
            X.append(torch.from_numpy(state))
            Y.append(y[i])
        cumuLoss = 0
        agent.train()
        for i in range(len(X)):

            pred = agent(X[i])
            loss = loss_fn(pred, Y[i].detach())
            cumuLoss += loss.item()

            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def updateTargetNetwork(self):
        self.Q_target = self.Q        

    def getPrediction(self, state, model):
        if np.random.rand() > EPSILON:
            with torch.no_grad():
                return torch.argmax(model(torch.from_numpy(np.array([state])))).item()
        return random.randrange(model.actionSpaceSize)

    def saveModel(self, agent):
        torch.save(agent.state_dict(), 'model_weights.pth')
        print("Model saved!")
    def loadModel(self, agent):
        agent.load_state_dict(torch.load("model_weights.pth"))
        print("Model loaded!")

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env.frameskip = 4
    state = env.reset()
    y = NeuralNetwork(env.action_space.n, state.shape[0]).to(device)
    target_y = NeuralNetwork(env.action_space.n, state.shape[0]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(y.parameters(), lr = learning_rate)
    agent = DQN()
    answer = input("Use a pre-trained model y/n? ")
    if answer == "y":
        agent.loadModel()
        EPSILON = 0.5697100000003659
    t = 0
    rewards = []
    episodes = []
    cumureward = 0
    for episode in range(EPISODES+500000000000):
        state = env.reset()
        cumureward = 0
        while True:
            env.render()
            cache = state.copy()
            action = agent.getPrediction(state,y)
            state, reward, done, info = env.step(action)
            agent.update_replay_memory((cache, action, reward, state, done))
            if len(agent.replay_memory) > 1000:
                agent.train(y, target_y, loss_fn, optimizer)
                EPSILON = max(EPSILON_MIN, EPSILON-EPSILON_DECAY)
                if t % 1000 == 0:
                    target_y.load_state_dict(y.state_dict())
            t+=1
            cumureward += reward
            if t % 10000 == 0:
                agent.saveModel(y)
            if done:
                break
        print("Score:", cumureward, " Episode:", episode, "Time:", t , " Epsilon:", EPSILON)