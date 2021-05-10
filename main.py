import sys
import gym
import envs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG import DDPGagent
from utils import *
import pudb
from tqdm import tqdm
import torch

from custom_gym import *

data=pd.read_csv('hvac_data.csv')

x=data.dropna(axis=0,how='any')
m=x.columns
p=x[x[m[3]]==max(x[m[3]])].index.values
len(p)
max(x[m[3]])
x=x.drop(p)

env = gym.make('CustomEnv-v0')
env.pass_df(x)
agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in tqdm(range(500)):
   
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(48):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if step == 499:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
# except:
#     pass

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("rewards.png")
    torch.save(agent.get_model().state_dict(), "models/"+str(episode)+".pth")
