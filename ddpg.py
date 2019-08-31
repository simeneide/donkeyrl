#%% IMPORTS
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import utils
from importlib import reload
reload(utils)

import datetime
def log(s):
    print(f"{datetime.datetime.now()}: {s}")

#%% MODEL
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.w1 = nn.Linear(1,10, bias = False)
        self.w2 = nn.Linear(10,2, bias = True)

    def forward(self, obs):
        x = torch.relu(self.w1(obs))
        x = self.w2(x)
        
        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.w1 = nn.Linear(obs_dim + action_dim,5, bias = True)
        self.w2 = nn.Linear(5,1, bias = True)
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.w1(x))
        x = self.w2(x)
        return x

## TEST
"""
critic = Critic(obs_dim = 1, action_dim = 2)
actor = Actor()

critic(state = batch['state'], action = batch['action'])
actor(batch['state'])
"""
#%%

class DDPGAgent:
    def __init__(self, headless=1):
        self.gamma = 0.99
        self.batch_size = 128
        self.critic_learning_rate = 0.005
        self.actor_learning_rate = 0.005

        self.tau = 0.001 # copy rate between target net and real net
        ## ENVIRONMENT
        self.env = utils.Denv(headless=headless, location = "donkey-generated-roads-v0")

        ## POLICY
        self.critic = Critic(obs_dim = 1, action_dim=2)
        self.critic_target = Critic(obs_dim = 1, action_dim=2)

        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = Actor()
        self.actor_target = Actor()
        # OPTIMIZERS
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        ## MEMORY
        self.memory = utils.ReplayBuffer(capacity=5000, seed = 0)

    def get_batch(self, bs = None):
        """Gather a batch from memory and collect into tensors"""
        if bs is None:
            bs = self.batch_size
        B = self.memory.sample_batch(bs)
        batch  = {
            'state' : torch.zeros((bs, 1)),
            'state_next' : torch.zeros((bs, 1)),
            'action' : torch.zeros((bs,2)),
            'done' : torch.zeros((bs,1)),
            'reward' : torch.zeros((bs,1))
        }
        
        for i, data in enumerate(B):
            batch['state'][i] = data.state
            batch['state_next'][i] = data.next_state
            batch['action'][i] = data.action
            batch['reward'][i] = data.reward
            batch['done'][i] = data.done

        return batch

    def play(self, explore = 0.05, print_reward=True):

        newobs = self.env.reset()

        random = torch.distributions.Normal(torch.zeros(1,2), explore)
        oh = random.sample()
        cumreward = 0.0
        
        for t in range(3000):
            curobs = newobs

            oh = oh*0.2 + random.sample()
            action = self.actor(curobs)+ oh
            action = action.clamp(-1.0,1.0)
            #action[:,1] = -0.5

            
            newobs, reward, done, info = self.env.step(action)
            cumreward += float(reward)

            self.memory.push(curobs,action,newobs,reward,done)
            if done:
                break

            if (t>300) & (info['speed'] < 1):
                break
        return {'reward' : cumreward, 'timesteps' : t}

    def update(self):
        with torch.no_grad():
            batch = self.get_batch()
        curr_Q = self.critic.forward(state = batch['state'], action = batch['action'])
        next_action = self.actor_target.forward(batch['state_next'])
        next_Q = self.critic_target.forward(state = batch['state_next'], action=next_action.detach())
        expected_Q = batch['reward'] + self.gamma * next_Q

        # Critic loss and step
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update actor
        policy_loss = -self.critic.forward(batch['state'], self.actor.forward(batch['state'])).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

policy = DDPGAgent(headless=0)

obs = policy.env.env.reset()

#%%

import matplotlib.pyplot as plt
plt.imshow(obs)

#%%
while len(policy.memory) < policy.batch_size:
    policy.play(explore = 0.5)

#%%

print("Start training..")
for i in range(100):
    for _ in range(20):
        policy.update()

    ## PLAY
    if i%5==0:
        exp = 0.0
        log("----- TEST RUN..")
    else:
        exp = torch.abs(0.6*torch.sin(torch.tensor(i/10)))
        
    res = policy.play(explore = exp)

    log(f"Iteration {i}: game over at t={res['timesteps']}. Tot reward : {res['reward']}. exp: {exp:.2f}")
#%%

#%%
