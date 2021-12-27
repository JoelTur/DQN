from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from tensorflow import keras
import random
import numpy as np
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE

##HYPERPARAMETERS
learning_rate = 0.0001
REPLAY_MEMORY_SIZE=100000
BATCH_SIZE = 64
GAMMA = 0.99
EPISODES = 5000
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = (EPSILON-EPSILON_MIN)/30000

class DQN:
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_model(self):
        model = Sequential()
        model.add(Dense(256,input_shape = (self.obsSpaceSize + 1,) ))
        model.add(Dense(256, activation = 'relu'))
        ##model.add(Dense(64, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mse', optimizer=adam_v2.Adam(lr=learning_rate), metrics = ['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        states = [transition[0] for transition in batch]
        next_states_jump = []
        next_states_skip = []
        for i, (state,action,reward,next_state, done) in enumerate(batch):
            cache_j = np.copy(next_state)
            cache_j = np.append(cache_j, 1)
            next_states_jump.append( cache_j )
            #np.delete(next_state, len(next_state) - 1)
            next_state = np.append(next_state, 0)
            next_states_skip.append( next_state )
            np.delete(next_state, len(next_state) - 1)
        
        y_jump = self.target_model.predict(np.array(next_states_jump))
        y_skip = self.target_model.predict(np.array(next_states_skip))
        X = []
        Y = np.zeros(BATCH_SIZE)
        #target_y = self.target_model.predict(np.array(next_states))
        
        for i,(state, action, reward, next_state, done) in enumerate(batch):
            if done:
                Y[i] = reward
            else:   
                Y[i] = reward + GAMMA*max(y_jump[i], y_skip[i])
        loss = self.model.train_on_batch(np.array(states), Y)

    def updateTargetNetwork(self):
        self.target_model.set_weights(self.model.get_weights())            

    def getPrediction(self, state):
        if np.random.rand() > EPSILON:
            cache_j = np.copy(state)
            cache_s = np.copy(state)
            cache_j = np.append(cache_j, 1)
            cache_s = np.append(cache_s, 0)
            jump = self.model.predict(np.array([cache_j]))
            skip = self.model.predict(np.array([cache_s]))
            if jump > skip: 
                return 1
            return 0
        return random.randrange(self.actionSpaceSize)

    def saveModel(self):
        self.model.save("/home/joel/Flappybird-DQN/bestmodel")
    def loadModel(self):
        self.model = keras.models.load_model("/home/joel/Flappybird-DQN/bestmodel/")

if __name__ == "__main__":
    game = FlappyBird(288,512,100)
    answer = input("Display screen y/n? ")
    display_screen = False
    if answer is "y":
        display_screen = True
    p = PLE(game, force_fps = True, frame_skip=2, display_screen=display_screen)
    p.init()
    state = np.array(list(p.getGameState().values()))
    agent = DQN(len(p.getActionSet()), state.shape[0])
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
        state = np.array(list(p.getGameState().values()))
        while True:
            action = agent.getPrediction(state)
            reward = p.act(p.getActionSet()[action])
            if p.game_over():
                reward = -1000
            next_state = np.array(list(p.getGameState().values()))
            state = np.append(state,action)
            agent.update_replay_memory((state, action, reward, next_state, p.game_over()))
            state = next_state
            if len(agent.replay_memory) > 1000:
                agent.train()
                EPSILON = max(EPSILON_MIN, EPSILON-EPSILON_DECAY)
                if t % 1000 == 0:
                    agent.updateTargetNetwork()
            t+=1
            cumureward += reward
            if t % 10000 == 0:
                agent.saveModel()
            if p.game_over():
                break
        print("Score:", cumureward, " Episode:", episode, "Time:", t , " Epsilon:", EPSILON)
