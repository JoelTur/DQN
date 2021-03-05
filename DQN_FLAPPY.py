from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE
learning_rate = 0.001
REPLAY_MEMORY_SIZE=2000
BATCH_SIZE = 32
GAMMA = 0.95
EPISODES = 5000
class DQN:
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.update = 0
    def create_model(self):
        model = Sequential()
        model.add(Dense(128,  input_shape = (self.obsSpaceSize,),activation ='relu' ))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(self.actionSpaceSize, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=Adam(lr=learning_rate))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, t_state):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        X = []
        Y = []
        for state, action, reward, next_state, done in batch:
            y = self.model.predict(np.array([state]))
            if done:
                y[0][action] = reward
            else:
                a = self.target_model.predict(np.array([next_state]))[0]
                y[0][action] = reward + GAMMA*np.amax(a)
            X.append(state)
            Y.append(y)
        self.model.fit(np.array(X), np.squeeze(np.array(Y)), epochs = 1, verbose = 0)
            #self.model.fit(np.array([state]), np.array(y), epochs = 1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if t_state == True:
            self.update += 1
        if self.update > 10:
            self.target_model.set_weights(self.model.get_weights())
            self.update = 0

    def getPrediction(self, state):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.model.predict(np.array([state])))
        ##print("RANDOM")
        return random.randrange(self.actionSpaceSize)

def processGameState(state):
    processed = np.array([])
    for element in state.values():
        if isinstance(element, dict):
            for ele in element.values():
                x = np.array(ele)
                if len(x.shape) == 1:
                    x = np.squeeze(x)
                else:
                    x = x.reshape(1, x.shape[0]*x.shape[1])
                processed = np.concatenate((processed, x), axis = None)
        else:
            processed = np.concatenate((processed, element), axis = None)
    return processed

if __name__ == "__main__":
    game = FlappyBird(288,512,100)
    p = PLE(game, fps = 30, display_screen=True)
    p.init()
    state = processGameState(p.getGameState())
    agent = DQN(len(p.getActionSet()), state.shape[0])
    i = 0
    t = 0
    for episode in range(EPISODES+500000000000):
        p.reset_game()
        #t = 0
        cumureward = 0
        state = processGameState(p.getGameState())
        while True:
            action = agent.getPrediction(state)
            reward = p.act(p.getActionSet()[action])
            if p.game_over() == True:
                reward = -1000
            next_state = processGameState(p.getGameState())
            agent.update_replay_memory((state, action, reward, next_state, p.game_over()))
            state = next_state
            if len(agent.replay_memory) > BATCH_SIZE:
                agent.train(p.game_over())
            t+=1
            if p.game_over() == True:
                break
        print("Time:", t, " Episode:", episode, " Epsilon:", agent.epsilon)
    