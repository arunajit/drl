import random
import gym
from gym import spaces
import numpy as np
import pandas as pd
from gym.utils import seeding
import math
import io
import torch
from copy import deepcopy
import scipy.io
import pudb
import json



max_temp_in=60
max_comfort=6
min_comfort=0
max_slots=3
max_steps=500
max_humid=100
min_humid=0
initial_temp=22

class CustomEnv(gym.Env):
    def __init__(self):
        self.seed()
        super(CustomEnv,self).__init__()
        self.reward_range=(-10,10)
        self.action_space= np.array([ 0,  1,  2])
        self.observation_space=spaces.Box(shape=(4,4),low=0,high=100,dtype=np.float32) 
        
    def pass_df(self,df):
        self.df=df
        print(self.df)
    
    def _next_observation(self):
        frame = np.array([np.array(self.df.loc[self.current_step: self.current_step + 3,'Air_temp'].values,dtype=np.float32),
                          np.array(self.df.loc[self.current_step: self.current_step + 3,'Relative_humidity'].values,dtype=np.float32),
                          np.array(self.df.loc[self.current_step: self.current_step + 3,'Outdoor_temp'].values,dtype=np.float32),
                          np.array(self.df.loc[self.current_step: self.current_step + 3,'Thermal_comfort'].values,dtype=np.float32)],dtype=np.float32)
        #obs=np.append(frame,[np.ndarray([int(self.temp)]),np.ndarray([self.humid]),np.ndarray([self.comfort]),np.ndarray([self.out_temp])],axis=0)
        obs=frame
        return obs
    
    def _take_action(self,action):
        current_temp=random.uniform(self.df.loc[self.current_step,'Air_temp'],self.df.loc[self.current_step,'Air_temp']+5)
        action_type=action[0]
        temperature=action[1]
        
        if action_type <1:
            self.temp +=5
            self.humid *= (self.temp/100)
            
        elif action_type <2:
            self.temp -=5
            self.humid *= (self.temp/100)
        
        self.comfort -=(self.temp*self.humid)/1000*6
        #self.out_temp=self.out_temp
        
        
    def step(self,action):
        self._take_action(action)
        self.current_step+=1
        if self.current_step > len(self.df.loc[:, 'Air_temp'].values) - 4:
            self.current_step = 0
        if self.comfort >4 and self.comfort <6:
            reward= self.comfort
        else:
            reward=-self.comfort
        done=self.comfort>4.5
        obs=self._next_observation()
        return obs, reward, done, {}
        
    
    def reset(self):
        self.temp=random.uniform(25,30)
        self.humid=random.uniform(30,40)
        self.comfort=random.uniform(2,5)
        #self.out_temp=random.uniform(28,35)
        self.current_step = random.randint(0, len(self.df.loc[:, 'Air_temp'].values) - 6)
        return self._next_observation()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] 
    
    def _render(self, mode='human', close=False):
        pass
