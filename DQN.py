from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import random
import numpy as np
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import skimage as skimage
from keras.regularizers import l2
from keras import initializers
from keras import callbacks
learning_rate = 0.000001
REPLAY_MEMORY_SIZE=50000
BATCH_SIZE = 32
GAMMA = 0.99
EPISODES = 5000
class DQN:
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
    def create_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, kernel_initializer='he_normal',kernel_regularizer=l2(0.01),subsample=(4, 4), border_mode='same',input_shape=(4,80,80)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, kernel_initializer='he_normal',kernel_regularizer=l2(0.01),subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, kernel_initializer='he_normal',kernel_regularizer=l2(0.01),subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(516, kernel_regularizer=l2(0.01),kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(2, kernel_regularizer=l2(0.01),kernel_initializer='he_normal'))
        model.add(Activation('linear'))
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, t):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        X = []
        Y = []
        for state, action, reward, next_state, done in batch:
            y = self.model.predict(np.array([state]))
            #print(y)
            if done:
                y[0][action] = reward
            else:
                a = self.target_model.predict(np.array([next_state]))[0]
                #print(a)
                y[0][action] = reward + GAMMA*np.amax(a)
            X.append(state)
            Y.append(y)
        #back = [callbacks.TensorBoard( write_grads= True)]
        his = self.model.fit(np.array(X), np.squeeze(np.array(Y)), epochs = 1, verbose = 0)
        #print(his.history["loss"])
            #self.model.fit(np.array([state]), np.array(y), epochs = 1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if t % 5000 == 0:
            self.target_model.set_weights(self.model.get_weights())

    def getPrediction(self, state, t):
        #if t % 4 != 0:
        #    return 1
        if np.random.rand() > self.epsilon:
            pred = self.model.predict(np.array([state]))
            #print(pred)
            #print(np.isnan(np.min(state)))
            return np.argmax(pred)
        return random.randrange(self.actionSpaceSize)
    
def getFrame(p):
    state = skimage.color.rgb2gray(p.getScreenRGB())
    state = skimage.transform.resize(state, (80,80))
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    return np.nan_to_num(state)/255.0

def updateFrameStack(p, state):
    state = np.delete(state, 0 , axis=0)
    update = skimage.color.rgb2gray(p.getScreenRGB())
    update = skimage.transform.resize(state, (80,80))
    update = skimage.exposure.rescale_intensity(update,out_range=(0,255))
    return np.append(state, update, axis = 0)


if __name__ == "__main__":
    game = FlappyBird(288,512,100)
    p = PLE(game, fps = 30, display_screen=True)
    p.init()
    #state = getFourFramesStacked(p)
    state = deque(maxlen = 4)
    agent = DQN(len(p.getActionSet()), None)
    i = 0
    t = 0
    totaltime = 0
    for episode in range(EPISODES+500000000000):
        #t = 0
        p.reset_game()
        #t = 0
        #state = getFourFramesStacked(p)
        state.append(getFrame(p))
        state.append(getFrame(p))
        state.append(getFrame(p))
        state.append(getFrame(p))
        while True:
            action = agent.getPrediction(state,t)
            reward = p.act(p.getActionSet()[action])
            oldstate = state.copy()
            state.append(getFrame(p))
            agent.update_replay_memory((oldstate, action, reward, state, p.game_over()))
            if len(agent.replay_memory) > BATCH_SIZE and t > 10000:
                agent.train(t)
            t+=1
            if p.game_over() == True:
                break
        print("Time:", t, " Episode:", episode, " Epsilon:", agent.epsilon)
