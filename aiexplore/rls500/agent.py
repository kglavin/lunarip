import torch
import random
import numpy as np
from collections import deque

from game import BallisticGameAI, Action, Point,ANGLE_FIRE_WOBBLE
from model import Linear_QNet, QTrainer
from helper import plot
import os
import math

#derived from https://github.com/python-engineer/snake-ai-pytorch

MAX_MEMORY = 1_000_000
BATCH_SIZE = 1000
LR = 0.001

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


action_list = [Action.A_UP,Action.A_DOWN,Action.FIRE]

state_info = [
            'angle',
            'range'
            ]

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 30 # randomness
        self.epsilon_max = 2
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.hint_memory = deque(maxlen=MAX_MEMORY)
        self.synthetic_memory = deque(maxlen=MAX_MEMORY//10)
        self.model = None
  
        file_name = os.path.join('./model', 'model.pth')
        if os.path.exists(file_name):
            self.model = torch.load(file_name)
            print("loaded")
        else:
            self.model = Linear_QNet(len(state_info), 12, len(onehot_action)) # first parm is the lenght of the state array 
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
            print(param_tensor, "\t", self.model.state_dict()[param_tensor])

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)



    def get_state(self, game): 
        angle = round(math.degrees(game.angle)-math.degrees(game.missile_alpha),3)
        #return np.array(state, dtype=float)
        state = [angle,game.missile_range]
        return np.array(state,dtype=float)

    def synthetic_data(self):
        for r in range(1,1200):
            state = [0, r]
            action = action_onehot[Action.FIRE]
            reward = 1600
            next_state = [0, r]
            done = False
            self.synthetic_memory.append((state, action, reward, next_state, done))

    def synthetic_train(self):
        if len(self.synthetic_memory) > 0:
            for c in range(1,3):
                if len(self.missile_synthetic_memory) > BATCH_SIZE//100:
                    mini_sample = random.sample(self.synthetic_memory, BATCH_SIZE//100)
                else:
                    mini_sample = random.sample(self.synthetic_memory, len(self.missile_synthetic_memory))
                states, actions, rewards, next_states, dones = zip(*mini_sample)
                self.trainer.train_step(states, actions, rewards, next_states, dones)


    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done)) 
    
    def hint_remember(self, state, action, reward, next_state, done):
        self.hint_memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_long_memory_hint(self):
        if len(self.hint_memory) > BATCH_SIZE:
            mini_sample = random.sample(self.hint_memory, BATCH_SIZE)
        else:
            mini_sample = srandom.sample(self.hint_memory, len(self.hint_memory))
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

 
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def _app_specific_hint(self,game):
        # instead of guessing provide a hint based on knowledge of the application
        return game.hint()

    def get_action(self, state):
        if (self.n_games % 10) == 0:
            self.epsilon_max += 1
        if random.randint(0, self.epsilon_max) < self.epsilon:
            move = random.choice(action_list)
            final_move = action_onehot[move]
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
            final_move = int_onehot[move]

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    total_reward = 0
    record = 0
    agent = Agent()
    game = BallisticGameAI()
    game_reward = 0
    episode = 250_000
    while episode > 0:
        episode -= 1
        hint = False
        # get old state
        state_old = agent.get_state(game)

        # get move
        
        # when enabled, gets a hint from the game, to speed up finding good moves.
        if random.randint(0,100) == 1: # disabled
            final_move = action_onehot[agent._app_specific_hint(game)]
            hint = True
        else:
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


        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if hint == True:
            agent.hint_remember(state_old, final_move, reward, state_new, done)
            hint = False

        if game.want_save == True:
            agent.model.save()
            game.want_save = False

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            #agent.train_long_memory_hint()


            total_reward += game_reward
            if score >= record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, "Game_Reward: ", game_reward, " Mem: ", len(agent.memory))
            game_reward = 0
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores,'Training')


def play(speed=60):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = BallisticGameAI(speed=speed)
    reward  = 0
    while True:
        # get old state
        state = agent.get_state(game)

        state0 = torch.tensor(state, dtype=torch.float32)
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


        #print("s: ",state[0],  " action: ", action, " ns: ", state_new[0])
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
            plot(plot_scores, plot_mean_scores,'Playing')

if __name__ == '__main__':
    train()
    play()