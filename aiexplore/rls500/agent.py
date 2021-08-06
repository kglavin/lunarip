
import torch
import random
import numpy as np
from collections import deque

from agenttypes import Action,onehot_action,action_onehot,int_onehot,action_list,state_info
from agenttypes import Point, StateStanza
from game import BallisticGameAI, AABattery,ANGLE_FIRE_WOBBLE,ANGLE_UP_SMALL,ANGLE_UP,ANGLE_DOWN_SMALL,ANGLE_DOWN
from model import Linear_QNet, QTrainer
from helper import plot,plot_init
import os
import math
import time
from termcolor import colored
import json



#derived from https://github.com/python-engineer/snake-ai-pytorch

MAX_MEMORY = 500_000
SYNTHETIC_MAX_MEMORY = 4_000_000
BATCH_SIZE = 1000
# random learning value with no hinting
LR = 0.0002
# full hinting.
LR = 0.00001
# full hinting after 1590 runs it got to 1.6350799082655783e-06 Nd rN out of episodes
LR = 0.001

class ScoringData:
    def __init__(self, data=None, game=None):
        self.data = data
        self.game = game

    def parse(self,d):
        if self.game is not None:
            return self.game.parse(d)
        else:
            return None,None,None,None,None

    def score(self,d):
        # if k is normalised then try variations
        k = self.parse(d)
        #otherwise do lookup
        if k is not None:
            return k, self.model_dict[k]
        else:
            return None,None

    def sample(self,number=0):
        if self.data is not None:
            keys = random.sample(self.data, number)
        else:
            keys = []
        return keys

class Agent:
    # decay ratio of 1.65 for full random
    # decay ratio of 1.05 for full hint.
    def __init__(self,lr=LR,filename='model.pth',decay_iterations=50_000, iter_growth_val = 1.07,ogamma=0.65): #was running at 0.7 
        self.n_games = 0
        self.epsilon = 240 # randomness
        self.epsilon_max = 2
        self.gamma = 0.9 # discount rate
        self.decay_iterations = decay_iterations
        self.iter_growth_val = iter_growth_val
        self.ogamma = ogamma
        self.memory = deque(maxlen=MAX_MEMORY)
        self.hint_memory = deque(maxlen=MAX_MEMORY)
        #self.synthetic_memory = deque(maxlen=SYNTHETIC_MAX_MEMORY)
        self.model = None
        self.game = None
        self.synthetic_data = None
        self.scoring_data = None
  
        file_name = os.path.join('./model', filename)
        if os.path.exists(file_name):
            self.model = torch.load(file_name)
            print("loaded")
        else:
            self.model = Linear_QNet(len(state_info), 16, len(onehot_action)) # first parm is the lenght of the state array 
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
            print(param_tensor, "\t", self.model.state_dict()[param_tensor])

        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma,decay_iterations=decay_iterations,iter_growth_val=iter_growth_val,ogamma=ogamma)




    def get_state(self, game): 
        return game.get_state()
        #angle = round(game.aa.alpha-game.target_alpha,3)
        ##return np.array(state, dtype=float)
        #state = [angle,game.target_range,game.aa.velocity]
        #return np.array(state,dtype=float)

    def add_fire_data(self,number=100,rng=1400):
        firedata = self.game.fire_data(rng)
        for s in random.choices(firedata,k=number):
            self.memory.append(s)
        smangle = self.game.small_angle_data(rng)
        for s in random.choices(smangle,k=number):
            self.memory.append(s)

        #if number > len(self.synthetic_memory):
        #    number = len(self.synthetic_memory)
        #for s in random.choices(self.synthetic_memory,k=number):
        #    self.memory.append(s)

    def synthetic_data(self):
        self.synthetic_data = game.synthetic_data()

    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done)) 
    
    def hint_remember(self, state, action, reward, next_state, done):
        self.hint_memory.append((state, action, reward, next_state, done)) 

    def train_long_memory_with_synthetic(self,discount=0.01):
        ## add in a set of synthetic data into the memory
        if len(self.synthetic_memory) > BATCH_SIZE:
            mini_sample = random.sample(self.synthetic_memory, BATCH_SIZE)
        else:
            mini_sample = random.sample(self.synthetic_memory, len(self.synthetic_memory)) 

        for idx in range(len(mini_sample)):
            state, action, reward, next_state, done = mini_sample[idx]
            if (action == action_onehot[Action.FIRE]) and (reward > 0):
                next_state = [round(random.randint(0,1572)/10000,3),
                                    random.randint(10,1200),self.game.aa.velocity]
                mini_sample[idx] = StateStanza(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        #print(f'training with len({len(dones)}) {states[0]}, {actions[0]})({states[1]}, {actions[1]})({states[2]}, {actions[2]})')
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #print("learning rate ",self.trainer.lr," ",self.trainer.decay_ratio," ",self.trainer.iterations)


    def train_long_memory(self):
        if len(self.memory) > 5*BATCH_SIZE:
            mini_sample = random.sample(self.memory, 5*BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_long_memory_hint(self):
        if len(self.hint_memory) > BATCH_SIZE:
            mini_sample = random.sample(self.hint_memory, BATCH_SIZE)
        else:
            mini_sample = random.sample(self.hint_memory, len(self.hint_memory))
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

 
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def _app_specific_hint(self,game):
        # instead of guessing provide a hint based on knowledge of the application
        return game.hint()

    def get_action_OLD(self, state):
        if (self.n_games > 20):
                #self.epsilon_max += 1
                self.epsilon_max = self.epsilon *10
            
        if random.randint(0, self.epsilon_max) < self.epsilon:
            move = random.choice(action_list)
            final_move = action_onehot[move]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
            final_move = int_onehot[move]
        return final_move

    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = int(torch.argmax(prediction).item())
        final_move = int_onehot[move]
        return final_move

    def get_model_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = int(torch.argmax(prediction).item())
        final_move = int_onehot[move]
        action = onehot_action[tuple(final_move)]
        #print(move,final_move,action)
        return action

    def model_sample(self, inputs):
        """ return a list that has a sample of actions based on state that the model would produce, inputs are the values that are queried from the model"""
        ret = []
        for sample in inputs:
            i,v = sample
            action = self.get_model_action([round(i,3), r,self.game.aa.velocity])
            ret.append((action,[round(i,3),r,self.game.aa.velocity]))
        return ret


    def model_describe(self):
        """ return a list of printable lines that allow a simple visualization of the state of the model"""
        cols = []
        for i in [ math.pi/2, math.pi/3,math.pi/4, math.pi/6, math.pi/8,math.pi/12, math.pi/16,math.pi/32, math.pi/64,math.pi/128, math.pi/256, math.pi/512,math.pi/720, 
                0, 
                -math.pi/720,-math.pi/512,-math.pi/256,-math.pi/128, -math.pi/64, -math.pi/32,-math.pi/16,-math.pi/12, -math.pi/8, -math.pi/6,-math.pi/4,  -math.pi/3,-math.pi/2]:
            a = []
            col = ""
            #for r in [50,100,200,300,400,500,600,700,900,1000,1100,1200,1300,1400]:
            for r in range(100,250,10):
                action = self.get_model_action([round(i,3), r,self.game.aa.velocity])
                a.append((action,[round(i,3),r,self.game.aa.velocity]))
            for t in a:
                action, state = t
                if action == Action.A_DOWN_LARGE:
                    col = col + colored('{:6s}','blue',attrs=['reverse']).format(str(state[1]))
                if action == Action.A_DOWN:
                    col = col + colored('{:6s}','blue').format(str(state[1]))
                if action == Action.FIRE:
                    col = col + colored('{:6s}','red').format(str(state[1]))
                if action == Action.A_UP_LARGE:
                    col = col + colored('{:6s}','green',attrs=['reverse']).format(str(state[1]))
                if action == Action.A_UP:
                    col = col + colored('{:6s}','green').format(str(state[1]))
            cols.append(('{:6s}'.format(str(round(i,3))),col))

        return cols

    def model_describe_print(self,episode):
            training = self.trainer.scheduler.state_dict()
            cols = self.model_describe() 
            #print("\033[F")
            #print("\033[F")

            #for a,b in cols:
            #    print("\033[F")
            print(self.n_games,episode,training['_last_lr']," ###############################################################")
            for i,col in cols:
                print(i,"\t",col)
            print("#######################################################################################")

    def save(self,filename):
            self.trainer.save(filename)

def train(lr=0.0001, episodes=300_000_000,ogamma=0.7,decay_iterations=40_000):
    plot_scores = []
    plot_mean_scores = []
    model_scores = []
    running_avg_model_scores = []
    learning_rates = []
    max_model_score = 0
    ravg_model_score = 0
    total_score = 0
    total_reward = 0
    record = 0
    agent = Agent(lr=lr,decay_iterations=decay_iterations,ogamma=ogamma)
    agent.game = game = BallisticGameAI()
    agent.synthetic_data = game.synthetic_data()
    agent.scoring_data = ScoringData(agent.synthetic_data,game)
    game_reward = 0
    episode = 0
    axes = plot_init('Training')

    #agent.add_fire_data(number=MAX_MEMORY)

    max_rnd = 1  #turnoff hinting from start
    used_hint = 0
    used_model = 0
    hint = True
    while episode < episodes:
        episode += 1

        # get old state
        state_old = agent.get_state(game)

        if hint == False: 
            if random.randint(1,max_rnd) == 1: 
                hint = True

        if hint == True:
            used_hint += 1
            final_move = action_onehot[agent._app_specific_hint(game)]
        else:
            used_model += 1
            final_move = agent.get_action(state_old)
        action = onehot_action[tuple(final_move)]

        # perform move and get new state
        reward, done, score,missile_hit = game.play_step(action)
        state_new = agent.get_state(game)
        game_reward += reward

        # only add a new missile after the new state has been recorded for use in training
        # add a new missile if needed
        if missile_hit == True:    
            game.find_missile()
            game.update_ui()
        else:   
            pass
            #update missile movement


        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if hint == True:
            agent.hint_remember(state_old, final_move, reward, state_new, done)
            hint = False

        if game.want_save == True:
            #agent.model.save()
            agent.save(filename='model')
            game.want_save = False

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            if (agent.n_games % 100) == 0:
                max_rnd +=1
                print(f'ngames = {agent.n_games},max_rnd = {max_rnd}')
            if agent.n_games < 40:
                hint = True
            else:
                hint = False

            #if agent.n_games % 2 == 0:
            #firedata = agent.n_games
            if 1 == 1:
                if agent.n_games ==  50:
                    firedata = 100
                if agent.n_games >= 50:
                    if firedata < 1000:
                        firedata += 1
                else:
                    firedata = 0
                print("Adding fire data: ", firedata)
                agent.add_fire_data(number=firedata,rng=250)
                
            agent.train_long_memory()
 
            total_reward += game_reward

            # attempt to score the model so that we have something better than score as the score is not good enough.
            model_score = 0
            samples = agent.scoring_data.sample(2500)
            for s in samples:
                state, action, _, _, _ = agent.scoring_data.parse(s)
                #check the model against the scoring data to see if they match.
                predicted_move = agent.get_action(state)
                #predicted_action = onehot_action[tuple(predicted_move)]
                if predicted_move == action:
                    model_score +=1
                
            if model_score > max_model_score:
                if model_score >  (max_model_score * 1.001):
                    agent.save(filename='max_model.pth.'+ str(model_score))
                    agent.model_describe_print(model_score)
                max_model_score = model_score

            if max_model_score == model_score or score >= record or (agent.n_games %100) == 0:
                record = score
                agent.save(filename='model.pth.'+ str(episode))
                #agent.model_describe()
            print(f'used_hint = {used_hint}, used_model = {used_model}')
            agent.model_describe_print(episode)
            #print('Game', agent.n_games, 'Score', score, 'Record:', record, "Game_Reward: ", game_reward, " Mem: ", len(agent.memory)," lr ",agent.trainer.lr, " Episode ", episode)
            game_reward = 0
            plot_scores.append(score)
            model_scores.append(model_score)
            ravg_model_score =  (4*ravg_model_score + model_score)//5
            running_avg_model_scores.append(ravg_model_score)
            
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            training = agent.trainer.scheduler.state_dict()
            learning_rates.append(training['_last_lr'])

            _=plot(axes,plot_scores, plot_mean_scores,model_scores,running_avg_model_scores,learning_rates)
    agent.save(filename='model.pth.'+ str(episode))


def play(speed=4,filename='model.pth'):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    record = 0
    agent = Agent(filename=filename)
    game = BallisticGameAI(speed=speed,iterations=3600)
    reward  = 0
    axes = plot_init('Playing')
    while True:
        # get old state
        state = agent.get_state(game)

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = agent.model(state0)
        move = int(torch.argmax(prediction).item())
        final_move = int_onehot[move]
        #final_move = action_onehot[agent._app_specific_hint(game)]
        action = onehot_action[tuple(final_move)]
        reward, done, score, missile_hit = game.play_step(action)
        state_new = agent.get_state(game)
        # add a new missile if needed
        if missile_hit == True:
            game.find_missile()
            game.update_ui()
            #print("missile destroyed")


        #print("s: ",state[0],state[1],  " action: ", action, " ns: ", state_new[0])
        if done:
            print("--------------------------------------------------------")
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(axes,plot_scores, plot_mean_scores,[],[],[])



if __name__ == '__main__':
    # angle,range, train(lr=0.001,episodes=600_000, ogamma=0.75)
    # angle,range,velocity,state
    #train(lr=0.001,episodetrain(lr=0.001,episodes=2_500_000, ogamma=0.89,decay_iterations=100_000)
    #train(lr=0.0001,episodes=2_500_000, ogamma=0.89,decay_iterations=100_000)
    train(lr=0.00001,episodes=2_500_000, ogamma=0.89,decay_iterations=300_000)
    

    #supertrain()
    #play(speed=120,filename='model.pth')
 