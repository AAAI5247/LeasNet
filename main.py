import os, sys, time
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import random

from buffer import ReplayBuffer
from keras.layers import Dense, Input, Flatten, merge
from keras.layers import Concatenate
from keras.models import Model
from keras.layers.core import Activation
from keras.engine.topology import Layer
from magent.builtin.tf_model import DeepQNetwork
from keras.callbacks import TensorBoard
import config
import make_env

def test(temp_steps):
    i_episode = 0
    n_step  = 2500
    max_steps = 500
    total_step = 0

    while total_step<n_step:
        alpha=0.01
        i_episode=i_episode+1
        steps = 0
        obs = env.reset()
        total_reward = 0
        obs = obs[0:n_agent]
        done = False
        while steps<max_steps:
            total_step += 1
            steps+=1
            ob=[]
            for j in range(n_agent):
                ob.append(np.asarray([obs[j]]))
            position_output = model_captain.predict(ob)
            position_output_selection = position_output.flatten()
            position = np.random.choice(n_agent, 1, p=position_output_selection)[0]
            combined_input = []
            for j in range(n_agent):
                combined_input.append(np.asarray([np.concatenate([ob[j][0],ob[position][0]], axis=0)]))
            model_combined_output_action = model_combined.predict(combined_input)
            model_independent_output_action = model_independent.predict(ob)

            action=np.zeros(n_agent+n_enemy,dtype = np.int32)
            for j in range(n_agent):
                if position == j:
                    if np.random.rand()<alpha:
                        action[j]=random.randrange(5)
                    else:
                        action[j]=np.argmax(model_independent_output_action[j])
                else:
                    if np.random.rand()<alpha:
                        action[j]=random.randrange(5)
                    else:
                        action[j]=np.argmax(model_combined_output_action[j])

            for j in range(n_enemy):
                action[j+n_agent]=random.randrange(action_dim)

            next_obs, reward, terminated,_= env.step(action)
            next_obs = next_obs[0:n_agent]
            done = (sum(terminated) > 0)
            obs = next_obs
            if(total_step == n_step):
                break
            if(done):
                break

    f = open('test.txt','a+')
    f.write(str(temp_steps) + '\t' + str(total_reward) + '\t'+  str(n_step/i_episode) + '\n')
    f.close()


def observation(state1,state2):
    state = []
    for j in range(n_agent):
        state.append(np.hstack(((state1[j][0:15,0:15,1]-state1[j][0:15,0:15,5]).flatten(),state2[j][-1:-3:-1])))
    return state

def Captain():
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

def cross_entropy(y, y_hat):
    assert y.shape == y_hat.shape
    res = -np.sum(np.nan_to_num(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
    return round(res, 3)

n_agent=5
n_enemy=1
np.random.seed(17)
action_dim = 5
observation_dim = 6

capacity = 200000
batch_size = 256
totalTime = 0
TAU = 0.01     
LRA = 0.0001        
param = None
alpha = 0.6
GAMMA = 0.96
n_episode = 2000
max_steps = 2500
episode_before_train = 50000
global_steps = 0

#environment configuration
env = make_env.make_env('predator_prey_obs')
sess = tf.Session()
K.set_session(sess)

i_episode=0
buff=ReplayBuffer(capacity)


#build the model
captain = Captain()
q_net_independent = Q_Net_Independent()
q_net_combined = Q_Net_Combined()

In= []
for j in range(n_agent):
    In.append(Input(shape=[observation_dim]))
In_pattern = merge(In,mode='concat',concat_axis=1) 
captain_out = captain(In)

V_independent = []

for j in range(n_agent):
    V_independent.append(q_net_independent(In[j]))

V_combined = []
In_combined = []
for j in range(n_agent):
    In_combined.append(Input(shape=[observation_dim*2]))
    V_combined.append(q_net_combined(In_combined[j]))

model_captain = Model(input=In,output=captain_out)
model_captain.compile(optimizer=Adam(lr = 0.00003), loss='categorical_crossentropy')

model_independent = Model(input=In,output=V_independent)
model_independent.compile(optimizer=Adam(lr = 0.00003), loss='mse')

model_combined = Model(input=In_combined,output=V_combined)
model_combined.compile(optimizer=Adam(lr = 0.00003), loss='mse')

#build the target model
captain_t = Captain()
q_net_independent_t = Q_Net_Independent()
q_net_combined_t = Q_Net_Combined()


In_t= []
for j in range(n_agent):
    In_t.append(Input(shape=[observation_dim]))
In_pattern_t = merge(In_t,mode='concat',concat_axis=1)

V_independent_t = []
for j in range(n_agent):
    V_independent_t.append(q_net_independent_t(In_t[j]))

captain_position_t = Input(shape = [observation_dim])
V_combined_t = []
In_combined_t = []
for j in range(n_agent):
    In_combined_t.append(Input(shape=[observation_dim*2]))
    V_combined_t.append(q_net_combined_t(In_combined_t[j]))

model_independent_t = Model(input=In_t,output=V_independent_t)
model_independent_t.compile(optimizer=Adam(lr = 0.00003), loss='mse')

model_combined_t = Model(input=In_combined_t,output=V_combined_t)
model_combined_t.compile(optimizer=Adam(lr = 0.00003), loss='mse')

start_time = time.time()
print(start_time)

while global_steps< 800000:
    alpha*=0.996
    if alpha<0.01:
        alpha=0.01
    i_episode=i_episode+1
    steps = 0
    obs = env.reset()
    total_reward = 0
    obs = obs[0:n_agent]
    done = False
    
    while steps<max_steps:
        steps+=1
        global_steps += 1
        i=0
        ob=[]
        for j in range(n_agent):
            ob.append(np.asarray([obs[j]]))
        position_output = model_captain.predict(ob)
        position_output_selection = position_output.flatten()
        position = np.random.choice(n_agent, 1, p=position_output_selection)[0]
        combined_input = []
        for j in range(n_agent):
            combined_input.append(np.asarray([np.concatenate([ob[j][0],ob[position][0]], axis=0)]))
        model_combined_output_action = model_combined.predict(combined_input)
        model_independent_output_action = model_independent.predict(ob)
        action=np.zeros(n_agent+n_enemy,dtype = np.int32)
        for j in range(n_agent):
            if position == j:
                if np.random.rand()<alpha:
                    action[j]=random.randrange(5)
                else:
                    action[j]=np.argmax(model_independent_output_action[j])
            else:
                if np.random.rand()<alpha:
                    action[j]=random.randrange(5)
                else:
                    action[j]=np.argmax(model_combined_output_action[j])

        for j in range(n_enemy):
            action[j+n_agent]=random.randrange(action_dim)
        next_obs, reward, terminated,_= env.step(action)
        reward = reward[0:n_agent]
        done = (sum(terminated) > 0)
        next_obs = next_obs[0:n_agent]
        ob_new = []
        for j in range(n_agent):
            ob_new.append(np.asarray([next_obs[j]]))
        position_output_new = model_captain.predict(ob_new)
        position_output_selection_new = position_output_new.flatten()
        position_new = np.random.choice(n_agent, 1, p=position_output_selection_new)[0]
        rewards = reward

        ce = 0
        for j in range(n_agent):
            if j == position:
                pass
            else:
                ce= ce+np.max(model_combined_output_action[j])-np.max(model_independent_output_action[j])

        advantages = np.zeros((1, n_agent))
        advantages[0][position] = ce
        model_captain.fit(ob,advantages, verbose=0)

        if steps%3 ==0:
            buff.add(obs, action[0:n_agent], reward, next_obs, done, position, position_new)
        
        obs = next_obs

        if(global_steps % 2500 ==0):
            test(global_steps)
        
        if(steps == 2500):
            done = True
        
        if(done):
            print(global_steps,total_reward,steps)
            f = open('train.txt','a+')
            f.write(str(global_steps) + '\t' + str(total_reward) + '\t'+  str(steps) + '\n')
            f.close()
            break

        if global_steps < 50000:
            continue
        if steps% 1 != 0:
            continue

        #training
        batch = buff.getBatch(128)
        Independent_states,actions,rewards,Independent_new_states,dones,positions,new_positions,Independent_states,Independent_new_states,Combined_states,Combined_new_states=[],[],[],[],[],[],[],[],[],[],[]
        for i_ in  range(n_agent):
            Independent_states.append([])
            Independent_new_states.append([])
            Combined_states.append([])
            Combined_new_states.append([])
        for e in batch:
            for j in range(n_agent):
                Independent_states[j].append(e[0][j])
                Combined_states[j].append(np.concatenate([e[0][j],e[0][e[5]]], axis=0))
                Independent_new_states[j].append(e[3][j])
                Combined_new_states[j].append(np.concatenate([e[3][j],e[0][e[6]]], axis=0))
            actions.append(e[1])
            rewards.append(e[2])
            dones.append(e[4])
            positions.append(e[5])
            new_positions.append(e[6])

        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        positions = np.asarray(positions)
        new_positions = np.asarray(new_positions)
        for i_ in  range(n_agent):
            Independent_states[i_]=np.asarray(Independent_states[i_])
            Independent_new_states[i_]=np.asarray(Independent_new_states[i_])
            Combined_states[i_]=np.asarray(Combined_states[i_])
            Combined_new_states[i_]=np.asarray(Combined_new_states[i_])
        independent_q_values = model_independent.predict(Independent_states)
        target_independent_q_values = model_independent_t.predict(Independent_new_states)
        Combined_q_values = model_combined.predict(Combined_states)
        target_Combined_q_values = model_combined_t.predict(Combined_new_states)
        for k in range(len(batch)):
            if dones[k]:
                for j in range(n_agent):
                    if j == positions[k]:

                        independent_q_values[j][k][actions[k][j]] = rewards[k][j]
            else:
                for j in range(n_agent):
                    if j == positions[k]:
                        independent_q_values[j][k][actions[k][j]] =rewards[k][j] + GAMMA*np.max(target_independent_q_values[j][k])

        history_1=model_independent.fit(Independent_states, independent_q_values,epochs=1,batch_size=128,verbose=0)

        for k in range(len(batch)):
            if dones[k]:
                for j in range(n_agent):
                    if j != positions[k]:

                        Combined_q_values[j][k][actions[k][j]] = rewards[k][j]
            else:
                for j in range(n_agent):
                    if j != positions[k]:

                        Combined_q_values[j][k][actions[k][j]] =rewards[k][j] + GAMMA*np.max(target_Combined_q_values[j][k])
        history_2=model_combined.fit(Combined_states, Combined_q_values,epochs=1,batch_size=128,verbose=0)

        #soft update
        weights = q_net_independent.get_weights()
        target_weights = q_net_independent_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        q_net_independent_t.set_weights(target_weights)

        weights = q_net_combined.get_weights()
        target_weights = q_net_combined_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        q_net_combined_t.set_weights(target_weights)

end_time = time.time()
print(end_time)
print(end_time-start_time)
