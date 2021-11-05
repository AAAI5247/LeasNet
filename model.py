import numpy as np
import magent
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import random
from keras.layers import Dense, Input, Flatten, merge
from keras.layers import Concatenate
from keras.models import Model
from keras.layers.core import Activation
from keras.engine.topology import Layer
from magent.builtin.tf_model import DeepQNetwork
from config import *
from keras.layers import *

def observation(state1,state2):
    state = []
    for j in range(n_agent):
        state.append(np.hstack(((state1[j][0:11,0:11,1]-state1[j][0:11,0:11,5]).flatten(),state2[j][-1:-3:-1])))
    return state

def Selection():
    input_list = []
    for i in range(n_agent):
        I1 = Input(shape = [observation_dim])
        input_list.append(I1)
    h = merge(input_list, mode='concat')
    h = Dense(512, activation='relu',kernel_initializer='random_normal')(h)
    h = Dense(128, activation='relu',kernel_initializer='random_normal')(h)
    V = Dense(n_agent, activation='softmax',kernel_initializer='random_normal')(h)
    model = Model(input=input_list,output=V)
    model.compile(optimizer=Adam(lr = 0.001), loss='categorical_crossentropy')
    return model

def Q_Net_Independent():
    I1 = Input(shape = [observation_dim])
    V = Dense(128,activation='relu',kernel_initializer='random_normal')(I1)
    V = Dense(64, activation='relu',kernel_initializer='random_normal')(V)
    V = Dense(action_dim, kernel_initializer='random_normal')(V)
    model = Model(input=I1,output=V)
    return model

def Q_Net_Combined():
    I1 = Input(shape = [observation_dim*2])
    V = Dense(128,activation='relu',kernel_initializer='random_normal')(I1)
    V = Dense(64, activation='relu',kernel_initializer='random_normal')(V)
    V = Dense(action_dim, kernel_initializer='random_normal')(V)
    model = Model(input=I1,output=V)
    return model
