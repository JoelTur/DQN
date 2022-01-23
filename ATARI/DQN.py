import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque

REPLAY_MEMORY_SIZE=2*10**5
BATCH_SIZE = 32
GAMMA = 0.99

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DQN(nn.Module):

    def __init__(self):
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.ddqn = True
        self.EPSILON = 0.1
        self.EPSILON_MIN = 0.1
        self.EPSILON_DECAY = (self.EPSILON-self.EPSILON_MIN)/250000

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, agent, target, loss_fn, optimizer):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        Y = []
        states = [torch.from_numpy(np.array(transition[0])/255) for transition in batch]
        states = torch.stack(states)
        states = states.float()
        next_states = [torch.from_numpy(np.array(transition[3])/255) for transition in batch]
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
            Y.append(y[i])
        Y = torch.stack(Y)
        agent.train()
        pred = agent(states)
        loss = loss_fn(pred, Y)
        loss.backward()
        for param in agent.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        self.EPSILON = max(self.EPSILON_MIN, self.EPSILON-self.EPSILON_DECAY)
        return loss.item()


    def getPrediction(self, state, model):
        if np.random.rand() > self.EPSILON:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                state = torch.from_numpy(state)
                state = state.float()
                return torch.argmax(model(state)).item()
        return random.randrange(model.actionSpaceSize)

    def saveModel(self, agent, filename):
        torch.save(agent.state_dict(), filename)
        print("Model saved!")
    def loadModel(self, agent, filename):
        agent.load_state_dict(torch.load(filename))
        print("Model loaded!")