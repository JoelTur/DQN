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
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0

    def push(self, transition: Tuple, error: float = None):
        if error is None:
            error = np.max(self.priorities) if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.memory.append(transition)
            self.size += 1
        else:
            self.memory[self.position] = transition

        self.priorities[self.position] = error ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        if self.size < self.capacity:
            probs = self.priorities[:self.size]
        else:
            probs = self.priorities

        probs = probs / np.sum(probs)
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error ** self.alpha

    def __len__(self) -> int:
        return self.size

class DQN(nn.Module):
    def __init__(self, replay_memory_size: int = 2*10**5, batch_size: int = 32, 
                 gamma: float = 0.99, epsilon: float = 1, epsilon_min: float = 0.1, 
                 epsilon_decay: int = 10**6, use_prioritized_replay: bool = False):
        super(DQN, self).__init__()
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.replay_memory = PrioritizedReplayBuffer(replay_memory_size)
        else:
            self.replay_memory = deque(maxlen=replay_memory_size)
            
        self.ddqn = True
        self.REPLAY_MEMORY_SIZE = replay_memory_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPSILON_MIN = epsilon_min
        self.EPSILON_DECAY = epsilon_decay
        self.last_save_time = 0
        self.save_interval = 300
        self.scaler = GradScaler()  # For mixed precision training

    def update_replay_memory(self, transition: Tuple):
        if self.use_prioritized_replay:
            self.replay_memory.push(transition)
        else:
            self.replay_memory.append(transition)

    def train(self, agent, target, loss_fn, optimizer):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return 0

        if self.use_prioritized_replay:
            batch, indices, weights = self.replay_memory.sample(self.BATCH_SIZE)
            weights = torch.FloatTensor(weights).to(device)
        else:
            batch = random.sample(self.replay_memory, self.BATCH_SIZE)
            weights = torch.ones(self.BATCH_SIZE).to(device)

        # Pre-allocate tensors for better performance
        states = torch.zeros((self.BATCH_SIZE, 4, 84, 84), dtype=torch.float32, device=device)
        next_states = torch.zeros((self.BATCH_SIZE, 4, 84, 84), dtype=torch.float32, device=device)
        actions = torch.zeros(self.BATCH_SIZE, dtype=torch.long, device=device)
        rewards = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=device)
        dones = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=device)

        # Batch process transitions
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            states[i] = torch.from_numpy(state/255).float()
            next_states[i] = torch.from_numpy(next_state/255).float()
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        # Optimize the model
        optimizer.zero_grad()

        # Use mixed precision training
        
        with autocast():
            # Get current Q values
            current_q_values = agent(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Get next Q values from target network

            with torch.no_grad():
                next_q_values = target(next_states)
                if self.ddqn:
                    next_actions = agent(next_states).max(1)[1]
                    next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
                else:
                    next_q_values = next_q_values.max(1)[0]

                target_q_values = rewards + (1 - dones) * self.GAMMA * next_q_values

            # Compute loss
            loss = (weights * loss_fn(current_q_values, target_q_values)).mean()


        # Backward pass with scaling and clipping
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)  # Critical for correct clipping
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            with torch.no_grad():
                errors = torch.abs(current_q_values - target_q_values).cpu().numpy()
                self.replay_memory.update_priorities(indices, errors)

        self.EPSILON = max(self.EPSILON_MIN, self.EPSILON-1/self.EPSILON_DECAY)
        return loss.item()

    def getPrediction(self, state, model):
        if np.random.rand() > self.EPSILON:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                state = torch.from_numpy(state).float().to(device)
                return torch.argmax(model(state)).item()
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