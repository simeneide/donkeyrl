import os
import argparse
import gym
import gym_donkeycar
import gym_donkeycar.envs
import time
import random
import numpy as np
from collections import  namedtuple, deque
from torch.utils.data import Dataset
import torch

#%% MEMORY
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer(object):

    def __init__(self, capacity, seed,
                 priority_weight=None, priority_exponent=None,
                 priotirized_experience=False):
        self.capacity = capacity
        self.position = 0
        self.prioritize = priotirized_experience
        self.priority_weight = priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent
        if self.prioritize:
            self.memory = []
        else:
            self.memory = []
        # Seed for reproducible results
        np.random.seed(seed)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)

#%% ENVIRONMENT

class Denv:
    def __init__(self, location = "donkey-generated-track-v0", headless=1):
        self.env = self.make_env(location , headless)
    
    def make_env(self, location , headless):
        os.environ['DONKEY_SIM_PATH'] = f"/Users/simeneide/Sync/donkey/donkey_sim.app/Contents/MacOS/donkey_sim" #args.sim" #f"/Users/simeneide/Sync/donkey/donkey_gym/donkey_sim.app/Contents/MacOS/donkey_sim" #args.sim
        rnd_port = np.random.randint(9000,9999)
        os.environ['DONKEY_SIM_PORT'] = str(rnd_port)
        os.environ['DONKEY_SIM_HEADLESS'] = str(headless)
        #%%
        env = gym.make(location)
        return env

    def step(self, action):
        action = action.flatten().detach().clone().numpy()
        action[1] = (action[1]+1)/2.0*5.0 # reshape throttle to be (0,5)

        obs, reward, done, info = self.env.step(action)
        
        # HACK: observation is now cte:
        obs = torch.tensor(info['cte']).view(1,-1)

        return obs, reward, done, info

    def reset(self):
        self.env.step(np.array([0.0,0.0]))
        self.env.reset()
        #time.sleep(0.5)
        self.env.reset()
        return torch.tensor(0.0).view(1,-1)

    def close(self):
        return self.env.close()

