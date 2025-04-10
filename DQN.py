import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
import os
from typing import List, Tuple, Optional
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.position = 0
        self.full = False
        self.observations = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((capacity), dtype=np.float32)
        self.rewards = np.zeros((capacity), dtype=np.float32)
        self.dones = np.zeros((capacity), dtype=np.bool)

    def size(self):
        if self.full:
            return self.capacity
        else:
            return self.position

            

    def push(self, observation: np.ndarray, next_observation: np.ndarray, action: int, reward: float, done: bool):
        self.observations[self.position] = np.array(observation).copy()
        if not done:
            self.observations[(self.position + 1) % self.capacity] = np.array(next_observation).copy()
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.position += 1
        if self.position == self.capacity:
            self.full = True
            self.position = 0

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.full:
            indices = (np.random.randint(1, self.capacity, size = batch_size) + self.position) % self.capacity
        else:
            indices = np.random.randint(0, self.position, size = batch_size)
        return self.observations[indices], self.observations[(indices + 1) % self.capacity], self.actions[indices], self.rewards[indices], self.dones[indices]


class DQN(nn.Module):
    def __init__(self, replay_memory_size: int = 2*10**5, batch_size: int = 32, 
                 gamma: float = 0.99, epsilon: float = 1, epsilon_min: float = 0.1, 
                 epsilon_decay: int = 10**6):
        super(DQN, self).__init__()
        self.replay_memory = ReplayBuffer(replay_memory_size)
            
        self.ddqn = False
        self.REPLAY_MEMORY_SIZE = replay_memory_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPSILON_MIN = epsilon_min
        self.EPSILON_DECAY = epsilon_decay
        self.last_save_time = 0
        self.save_interval = 300


    def update_replay_memory(self, observation: np.ndarray, next_observation: np.ndarray, action: int, reward: float, done: bool):
        self.replay_memory.push(observation, next_observation, action, reward, done)

    def train(self, agent, target, loss_fn, optimizer):



        observations, next_observations, actions, rewards, dones = self.replay_memory.sample(self.BATCH_SIZE)


        states = torch.from_numpy(observations).float().to(device)
        next_states = torch.from_numpy(next_observations).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)


        # Use mixed precision training
        

        # Get current Q values
        current_q_values = agent(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()



        # Get next Q values from target network
         
        with torch.no_grad():
            if self.ddqn:
                next_actions = agent(next_states).max(1)[1]
                next_q_values = target(next_states)
                next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q_values = target(next_states)
                next_q_values = next_q_values.max(1)[0]

            target_q_values = rewards + (1 - dones.float()) * self.GAMMA * next_q_values
    
        # Compute loss
        loss = loss_fn(current_q_values, target_q_values).mean()
        
        # Optimize the model
        optimizer.zero_grad()

        # Backward pass with scaling and clipping
        loss.backward()

        for param in agent.parameters():
            param.grad = torch.clamp(param.grad, -1, 1)
        optimizer.step()
        


        return loss.item(), current_q_values.max().item(), current_q_values.mean().item()
    
    def reduce_epsilon(self):
        self.EPSILON = max(self.EPSILON_MIN, self.EPSILON-1/self.EPSILON_DECAY)

    def pred(self, state, model):
        with torch.no_grad():
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(device)
            return torch.argmax(model(state)).item()
    
    
    def epsilon_greedy(self, state, model):
        if np.random.rand() > self.EPSILON:
            return self.pred(state, model)
        return random.randrange(model.actionSpaceSize)
    


    def saveModel(self, agent, filename):
        try:
            torch.save(agent.state_dict(), filename)
            print("Model saved!")
        except Exception as e:
            print(f"Error saving model: {e}")

    def loadModel(self, agent, filename):
        try:
            agent.load_state_dict(torch.load(filename), strict=False)
            print(f"Model loaded from {filename}!")
        except Exception as e:
            print(f"Error loading model: {e}")