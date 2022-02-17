import skimage
import numpy as np
import hyperparameters

def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0

def getFrame(x):
    #x = x[35:210,0:160]## For breakout, pong
    #x = x[0:405,0:288] #flappybird
    x = x[75:170,10:160]#For robotank :D
    state = skimage.color.rgb2gray(x)
    state = skimage.transform.resize(state, hyperparameters.INPUTSIZE)
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    state = state.astype('uint8')
    return state

def makeState(state):
    return np.stack((state[0],state[1],state[2],state[3]), axis=0)