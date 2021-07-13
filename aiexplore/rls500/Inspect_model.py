import torch
import random
import numpy as np
from collections import deque
from game import BallisticGameAI, Action, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import math
from termcolor import colored
import sys


onehot_action = { 
    (1,0,0,0,0): Action.A_UP,
    (0,1,0,0,0): Action.A_DOWN,
    (0,0,1,0,0): Action.FIRE,
    (0,0,0,1,0): Action.A_UP_LARGE,
    (0,0,0,0,1): Action.A_DOWN_LARGE
}

action_onehot = { 
    Action.A_UP:   [1,0,0,0,0],
    Action.A_DOWN: [0,1,0,0,0],
    Action.FIRE:   [0,0,1,0,0], 
    Action.A_UP_LARGE:   [0,0,0,1,0],
    Action.A_DOWN_LARGE:   [0,0,0,0,1],
}
int_onehot = {
    0:   [1,0,0,0,0],
    1:   [0,1,0,0,0],
    2:   [0,0,1,0,0],
    3:   [0,0,0,1,0],
    4:   [0,0,0,0,1],

}


action_list = [Action.A_UP,Action.A_DOWN,Action.FIRE,Action.A_UP_LARGE,Action.A_DOWN_LARGE]

state_info = [
            'angle',
            'range'
            ]

class Agent:
    def __init__(self,filename='model.pth'):
        file_name = os.path.join('/Users/kevin/GitHub/lunarip/aiexplore/rls500/model', filename)
        if os.path.exists(file_name):
            self.model = torch.load(file_name)
            print("loaded")
        else:
            self.model = Linear_QNet(len(state_info), 16, len(onehot_action)) # first parm is the lenght of the state array 
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
            print(param_tensor, "\t", self.model.state_dict()[param_tensor])
    
    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float32)
        prediction = self.model(state0)
        move = int(torch.argmax(prediction).item())
        final_move = int_onehot[move]
        action = onehot_action[tuple(final_move)]
        #print(move,final_move,action)
        return action
 


if __name__ == '__main__':
    #print(sys.argv[1])
    #agent = Agent(filename=sys.argv[1])
    agent = Agent()
    good=0
    bad=0

    for i in [ math.pi/2, math.pi/3,math.pi/4, math.pi/6, math.pi/8,math.pi/12, math.pi/16,math.pi/32, math.pi/64,math.pi/128, math.pi/256, math.pi/512, 
                0, 
                -math.pi/512,-math.pi/256,-math.pi/128, -math.pi/64, -math.pi/32,-math.pi/16,-math.pi/12, -math.pi/8, -math.pi/6,-math.pi/4,  -math.pi/3,-math.pi/2]:
        a = []
        col = ""
        for r in [50,100,200,300,400,500,600,700,900,1000,1100,1200]:
            action = agent.get_action([round(i,3), r])
            a.append((action,[round(i,3),r]))
        for t in a:
            action, state = t
            if action == Action.A_DOWN_LARGE:
                col = col + colored(state,'blue',attrs=['reverse'])
            if action == Action.A_DOWN:
                col = col + colored(state,'blue')
            if action == Action.FIRE:
                col = col + colored(state,'red')
            if action == Action.A_UP_LARGE:
                col = col + colored(state,'green',attrs=['reverse'])
            if action == Action.A_UP:
                col = col + colored(state,'green')
        print(round(i,3),"\t",col)

