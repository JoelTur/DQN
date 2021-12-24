from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
import random
import numpy as np
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE

learning_rate = 0.001
REPLAY_MEMORY_SIZE=100000
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
    def create_model(self):
        model = Sequential()
        model.add(Dense(64,input_shape = (self.obsSpaceSize,) ))
        model.add(Dense(64, activation = 'relu'))
        ##model.add(Dense(128, activation = 'relu'))
        model.add(Dense(self.actionSpaceSize, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=adam_v2.Adam(lr=learning_rate))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        states = [transition[0] for transition in batch]
        next_states = [transition[3] for transition in batch]
        X = []
        Y = []
        y = self.model.predict(np.array(states))
        target_y = self.target_model.predict(np.array(next_states))
        for i,(state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            else:
                y[i][action] = reward + GAMMA*np.amax(target_y[i])
            X.append(state)
            Y.append(y[i])
        ##print(np.array(Y))
        self.model.train_on_batch(np.array(X), np.array(Y))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def updateTargetNetwork():
            self.target_model.set_weights(self.model_get_weights())            

    def getPrediction(self, state):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.model.predict(np.array([state])))
        return random.randrange(self.actionSpaceSize)

    def saveModel(self):
        self.model.save("/home/joel/Flappybird-DQN/bestmodel")

if __name__ == "__main__":
    game = FlappyBird(288,512,100)
    p = PLE(game, force_fps = True, frame_skip=3, display_screen=True)
    p.init()
    state = np.array(list(p.getGameState().values()))
    agent = DQN(len(p.getActionSet()), state.shape[0])
    t = 0
    rewards = []
    episodes = []
    cumureward = 0
    for episode in range(EPISODES+500000000000):
        p.reset_game()
        cumureward = 0
        state = np.array(list(p.getGameState().values()))
        while True:
            action = agent.getPrediction(state)
            reward = p.act(p.getActionSet()[action])
            next_state = np.array(list(p.getGameState().values()))
            agent.update_replay_memory((state, action, reward, next_state, p.game_over()))
            state = next_state
            if len(agent.replay_memory) > BATCH_SIZE:
                agent.train()
                if t % 2000 == 0:
                    agent.updateTargetNetwork()
            t+=1
            cumureward += reward
            if p.game_over() == True and t % 10000:
                agent.saveModel()
                break
        print("Score:", cumureward, " Episode:", episode, " Epsilon:", agent.epsilon)
