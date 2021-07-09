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


onehot_action = { 
    (1,0,0): Action.A_UP,
    (0,1,0): Action.A_DOWN,
    (0,0,1): Action.FIRE
}

action_onehot = { 
    Action.A_UP:   [1,0,0],
    Action.A_DOWN: [0,1,0],
    Action.FIRE:   [0,0,1], 
}
int_onehot = {
    0:   [1,0,0],
    1:   [0,1,0],
    2:   [0,0,1]
}


action_list = [Action.A_UP,Action.A_DOWN,
            Action.FIRE]

state_info = [
            'angle',
            'range'
            ]

class Agent:
    def __init__(self):
        file_name = os.path.join('./model', 'model.pth')
        if os.path.exists(file_name):
            self.model = torch.load(file_name)
            print("loaded")
        else:
            self.model = Linear_QNet(len(state_info), 12, len(onehot_action)) # first parm is the lenght of the state array 
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
    agent = Agent()
    good=0
    bad=0

#    for i in [ -1, 0, 1]:
#        state = 'Nil'
#        action = agent.get_action([i])
#
#        if (action == Action.A_DOWN) and i == 1:
#            state = ' Good'
#            good+=1
#        elif (action == Action.FIRE) and i == 0:
#            state = ' Good'
#            good+=1
#        elif (action == Action.A_UP) and i == -1:
#            state = ' Good'
#            good+=1
#        else:
#            state = ' Bad'
#            bad +=1

    for i in range(-50,50):
        a = []
        col = ""
        for r in [100,200,300,400,500,600,700,900,1000]:
            action = agent.get_action([i, r])
            a.append((action,[i,r]))
        for t in a:
            action, state = t
            if action == Action.A_DOWN:
                col = col + colored(state,'blue')
            if action == Action.FIRE:
                col = col + colored(state,'red')
            if action == Action.A_UP:
                col = col + colored(state,'green')
        print(i,col)

    #bg_string = "g: " +str(good) + " b: " + str(bad) + " b/g: " + str(bad/(good+0.01))
    #print(bg_string)
    #with open('model.status.bg', 'a') as the_file:
    #    the_file.write(bg_string+'\n')

