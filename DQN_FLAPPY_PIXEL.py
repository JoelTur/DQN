import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE
import skimage
from torch.nn import functional as F

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
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(6400, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, self.actionSpaceSize)

    def forward(self, x):
        x = x.to(device)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


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
        states = [torch.from_numpy(np.array(transition[0])) for transition in batch]
        states = torch.stack(states)
        states = states.float()
        next_states = [torch.from_numpy(np.array(transition[3])) for transition in batch]
        next_states = torch.stack(next_states)
        next_states = next_states.float()
        optimizer.zero_grad()
        y = agent(states)
        target_y = target(next_states)
        y_next = agent(next_states)
        for i,(state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            elif self.ddqn is False:
                y[i][action] = reward + GAMMA*torch.max(target_y[i])
            else:
                y[i][action] = reward + GAMMA*target_y[i][torch.argmax(y_next[i])]
            X.append(torch.from_numpy(state))
            Y.append(y[i])
        Y = torch.stack(Y)
        agent.train()
        pred = agent(states)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()      

    def getPrediction(self, state, model):
        if np.random.rand() > EPSILON:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                state = torch.from_numpy(state)
                state = state.float()
                return torch.argmax(model(state)).item()
        return random.randrange(model.actionSpaceSize)

    def saveModel(self, agent):
        torch.save(agent.state_dict(), 'pixel_flappy_weights.pth')
        print("Model saved!")
    def loadModel(self, agent):
        agent.load_state_dict(torch.load("pixel_flappy_weights.pth"))
        print("Model loaded!")
def getFrame(p):
    state = skimage.color.rgb2gray(p.getScreenRGB())
    state = skimage.transform.resize(state, (80,80))
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    return state/255

def makeState(state):
    return np.stack((state[0],state[1],state[2],state[3]), axis=0)

if __name__ == "__main__":
    game = FlappyBird(288,512,100)
    answer = input("Display screen y/n? ")
    display_screen = False
    if answer is "y":
        display_screen = True
    p = PLE(game, force_fps = True, frame_skip=3, display_screen=display_screen)
    p.init()
    y = NeuralNetwork(len(p.getActionSet()), None).to(device)
    target_y = NeuralNetwork(len(p.getActionSet()), None).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(y.parameters(), lr = learning_rate)
    agent = DQN()
    state = deque(maxlen = 4)
    print(y)
    answer = input("Use a pre-trained model y/n? ")
    if answer is "y":
        agent.loadModel()
        EPSILON = 0.1
    t = 0
    rewards = []
    episodes = []
    cumureward = 0
    for episode in range(EPISODES+500000000000):
        p.reset_game()
        cumureward = 0
        state.append(getFrame(p))
        state.append(getFrame(p))
        state.append(getFrame(p))
        state.append(getFrame(p))
        while True:
            action = agent.getPrediction(makeState(state),y)
            reward = p.act(p.getActionSet()[action])
            cache = state.copy()
            state.append(getFrame(p))
            agent.update_replay_memory((makeState(cache), action, reward, makeState(state), p.game_over()))
            if len(agent.replay_memory) > 1000:
                agent.train(y, target_y, loss_fn, optimizer)
                EPSILON = max(EPSILON_MIN, EPSILON-EPSILON_DECAY)
                if t % 1000 == 0:
                    target_y.load_state_dict(y.state_dict())
            t+=1
            cumureward += reward
            if t % 10000 == 0:
                agent.saveModel(y)
            if p.game_over():
                break
        print("Score:", cumureward, " Episode:", episode, "Time:", t , " Epsilon:", EPSILON)
