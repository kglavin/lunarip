import torch
import random
import numpy as np
from collections import deque
from game import BallisticGameAI, Action, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

#derived from https://github.com/python-engineer/snake-ai-pytorch

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

onehot_action = { 
    (1,0,0,0,0,0,0,0,0): Action.V_UP,
    (0,1,0,0,0,0,0,0,0): Action.V_DOWN,
    (0,0,1,0,0,0,0,0,0): Action.A_UP,
    (0,0,0,1,0,0,0,0,0): Action.A_DOWN,
    (0,0,0,0,1,0,0,0,0): Action.FIRE,
    (0,0,0,0,0,1,0,0,0): Action.V_UP10,
    (0,0,0,0,0,0,1,0,0): Action.V_DOWN10,
    (0,0,0,0,0,0,0,1,0): Action.A_UP10,
    (0,0,0,0,0,0,0,0,1): Action.A_DOWN10
}

action_onehot = { 
    Action.V_UP:   [1,0,0,0,0,0,0,0,0],
    Action.V_DOWN: [0,1,0,0,0,0,0,0,0],
    Action.A_UP:   [0,0,1,0,0,0,0,0,0],
    Action.A_DOWN: [0,0,0,1,0,0,0,0,0],
    Action.FIRE:   [0,0,0,0,1,0,0,0,0],
    Action.V_UP10: [0,0,0,0,0,1,0,0,0],
    Action.V_DOWN10: [0,0,0,0,0,0,1,0,0],
    Action.A_UP10:   [0,0,0,0,0,0,0,1,0],
    Action.A_DOWN10: [0,0,0,0,0,0,0,0,1] 
}
int_onehot = { 
    0:   [1,0,0,0,0,0,0,0,0],
    1:   [0,1,0,0,0,0,0,0,0],
    2:   [0,0,1,0,0,0,0,0,0],
    3:   [0,0,0,1,0,0,0,0,0],
    4:   [0,0,0,0,1,0,0,0,0],
    5:   [0,0,0,0,0,1,0,0,0],
    6:   [0,0,0,0,0,0,1,0,0],
    7:   [0,0,0,0,0,0,0,1,0],
    8:   [0,0,0,0,0,0,0,0,1] 
}

action_list = [Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP10,Action.A_DOWN10,
            Action.FIRE,
            Action.FIRE,
            Action.A_UP,Action.A_DOWN,
            Action.A_UP10,Action.A_DOWN10,
            Action.A_UP,Action.A_DOWN]

action_list = [Action.A_UP,Action.A_DOWN,
            Action.A_UP10,Action.A_DOWN10,
            Action.FIRE]

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None
  
        file_name = os.path.join('./model', 'model.pth')
        if os.path.exists(file_name):
            self.model = torch.load(file_name)
        else:
            self.model = Linear_QNet(6, 64, 9)


        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game): 
        state = [
            game.angle*10,
            game.velocity,
            game.missile_loc.x+game.missile_size//2,
            game.missile_loc.y+game.missile_size//2,
            game.missile_alpha,
            game.shots
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.choice(action_list)
            final_move = action_onehot[move]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = int_onehot[move]

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = BallisticGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        action = onehot_action[tuple(final_move)]

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if game.want_save == True:
            agent.model.save()
            game.want_save = False

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()