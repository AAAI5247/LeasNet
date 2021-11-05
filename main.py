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
from model import *
from config import *
from buffer import ReplayBuffer
from keras.layers import *

#environment configuration
magent.utility.init_logger("battle")
env = magent.GridWorld("battle", map_size=20)
env.set_render_dir("./build/render")
handles = env.get_handles()
sess = tf.Session()
K.set_session(sess)
n = len(handles)
n_actions=env.get_action_space(handles[0])[0]
i_episode=0
buff=ReplayBuffer(capacity)


#build the model
selection = Selection()
q_net_independent = Q_Net_Independent()
q_net_combined = Q_Net_Combined()

In= []
for j in range(n_agent):
    In.append(Input(shape=[observation_dim]))
In_pattern = merge(In,mode='concat',concat_axis=1) 
selection_out = selection(In)

V_independent = []

for j in range(n_agent):
    V_independent.append(q_net_independent(In[j]))

V_combined = []
In_combined = []
for j in range(n_agent):
    In_combined.append(Input(shape=[observation_dim*2]))
    V_combined.append(q_net_combined(In_combined[j]))

model_selection = Model(input=In,output=selection_out)
model_selection.compile(optimizer=Adam(lr = 0.00003), loss='categorical_crossentropy')

model_independent = Model(input=In,output=V_independent)
model_independent.compile(optimizer=Adam(lr = 0.00003), loss='mse')

model_combined = Model(input=In_combined,output=V_combined)
model_combined.compile(optimizer=Adam(lr = 0.00003), loss='mse')

#build the target model
selection_t = Selection()
q_net_independent_t = Q_Net_Independent()
q_net_combined_t = Q_Net_Combined()


In_t= []
for j in range(n_agent):
    In_t.append(Input(shape=[observation_dim]))
In_pattern_t = merge(In_t,mode='concat',concat_axis=1)

V_independent_t = []
for j in range(n_agent):
    V_independent_t.append(q_net_independent_t(In_t[j]))

selection_position_t = Input(shape = [observation_dim])
V_combined_t = []
In_combined_t = []
for j in range(n_agent):
    In_combined_t.append(Input(shape=[observation_dim*2]))
    V_combined_t.append(q_net_combined_t(In_combined_t[j]))

model_independent_t = Model(input=In_t,output=V_independent_t)
model_independent_t.compile(optimizer=Adam(lr = 0.00003), loss='mse')

model_combined_t = Model(input=In_combined_t,output=V_combined_t)
model_combined_t.compile(optimizer=Adam(lr = 0.00003), loss='mse')

tf_model = DeepQNetwork(env, handles[1], 'trusty-battle-game-l', use_conv=True)
tf_model.load("data/battle_model", 0, 'trusty-battle-game-l')

while i_episode<n_episode:
    alpha*=0.996
    if alpha<0.01:
        alpha=0.01
    print(i_episode)
    i_episode=i_episode+1
    env.reset()
    env.add_agents(handles[0], method="random", n=n_agent)
    env.add_agents(handles[1], method="random", n=n_enemy)
    done = False
    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    action = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    steps = 0
    score = 0
    loss = 0
    dead = [0,0]
    
    while steps<max_steps:
        steps+=1
        i=0
        obs[i] = env.get_observation(handles[i])
        flat_ob = observation(obs[i][0],obs[i][1])
        ob=[]
        for j in range(n_agent):
            ob.append(np.asarray([flat_ob[j]]))
        position_output = model_selection.predict(ob)
        position_output_selection = position_output.flatten()
        position = np.random.choice(n_agent, 1, p=position_output_selection)[0]

        combined_input = []
        for j in range(n_agent):
            combined_input.append(np.asarray([np.concatenate([ob[j][0],ob[position][0]], axis=0)]))
        model_combined_output_action = model_combined.predict(combined_input)
        model_independent_output_action = model_independent.predict(ob)
        action[i]=np.zeros(n_agent,dtype = np.int32)
        for j in range(n_agent):
            if position == j:
                if np.random.rand()<alpha:
                    action[i][j]=random.randrange(n_actions)
                else:
                    action[i][j]=np.argmax(model_independent_output_action[j])
            else:
                if np.random.rand()<alpha:
                    action[i][j]=random.randrange(n_actions)
                else:
                    action[i][j]=np.argmax(model_combined_output_action[j])

        env.set_action(handles[i], action[i])
        obs[1] = env.get_observation(handles[1])
        ids[1] = env.get_agent_id(handles[1])
        acts = tf_model.infer_action(obs[1], ids[1], 'e_greedy')
        env.set_action(handles[1], acts)
        done = env.step()
        next_obs = env.get_observation(handles[0])
        flat_next_obs = observation(next_obs[0],next_obs[1])
        ob_new = []
        for j in range(n_agent):
            ob_new.append(np.asarray([flat_next_obs[j]]))
        position_output_new = model_selection.predict(ob_new)
        position_output_selection_new = position_output_new.flatten()
        position_new = np.random.choice(n_agent, 1, p=position_output_selection_new)[0]
        rewards = env.get_reward(handles[0])
        score += sum(rewards)

        ce = 0
        for j in range(n_agent):
            if j == position:
                pass
            else:
                ce= ce+np.max(model_combined_output_action[j])-np.max(model_independent_output_action[j])
        advantages = np.zeros((1, n_agent))
        advantages[0][position] = ce
        model_selection.fit(ob,advantages, verbose=0)

        if steps%3 ==0:
            buff.add(flat_ob, action[0], rewards, flat_next_obs, done, position, position_new)

        env.clear_dead()

        #add agents
        idd = n_agent - len(env.get_agent_id(handles[0]))
        if idd>0:
            env.add_agents(handles[0], method="random", n=idd)
            dead[0]+=idd
        idd = n_enemy - len(env.get_agent_id(handles[1]))
        if idd>0:
            env.add_agents(handles[1], method="random", n=idd)
            dead[1]+=idd

        if i_episode < episode_before_train:
            continue
        if steps%3 != 0:
            continue


        #training
        batch = buff.getBatch(batch_size)
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

