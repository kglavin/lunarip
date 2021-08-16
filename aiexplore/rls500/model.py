
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json

from hyper import BATCH_SIZE

#derived from https://github.com/python-engineer/snake-ai-pytorch

class Linear_QNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

    def save(self, file_name='model.pth'):
        torch.save(self, file_name)

class QTrainer:
    def __init__(self, model, lr, gamma,decay_iterations=100, iter_growth_val = 1.1,ogamma=0.7):
        self.lr = lr
        self.gamma = gamma
        self.decay_iterations = decay_iterations
        self.iter_growth_val = iter_growth_val
        self.decay_steps = 0
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=ogamma, verbose=True)
        self.criterion = nn.MSELoss()
        self.iterations = 0

    def train_step2(self, state, action, reward, next_state, done):
        BATCH_SIZE = 1  # from agent.py
        self.iterations += 1
        if self.iterations*BATCH_SIZE > self.decay_iterations:
            self.iterations = 0 
            self.decay_iterations = self.decay_iterations + (BATCH_SIZE * self.decay_steps) * self.iter_growth_val
            self.decay_steps += 1
            if self.decay_steps < 5:
                if self.decay_iterations >40000: #was 500K
                    self.decay_iterations = 40000
            else:
                scheduled_lr = self.scheduler.state_dict()['_last_lr']
                self.decay_iterations = 40_000 + 0.5/float(scheduled_lr[0])
                print("adding to exponential decay iteraions ",scheduled_lr,self.decay_iterations)
            print(" ************************************ Scheduler.step() *****************************", )
            self.scheduler.step()
            print(" ************************************")
            print(self.scheduler.state_dict())
            print(" ************************************ Scheduler.step() COMPLETE *********************", )
        else:
            if self.iterations % 100000 == 0:
                print("Iteration ",self.iterations, " Batch_iteration ", self.iterations*BATCH_SIZE, " of Decay Iterations ", self.decay_iterations)
                
            
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int32)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                ns = next_state[idx]
                a = self.model(ns)
                Q_new = reward[idx] + self.gamma * torch.max(a)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def train_step(self, state, action, reward, next_state, done):
             
        batch_step = True

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int32)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            batch_step = False
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        

        if batch_step == True:
            self.iterations += 1
            if self.iterations > self.decay_iterations:
                self.iterations = 0 
                self.decay_iterations = self.decay_iterations * self.iter_growth_val
                self.decay_steps += 1
                if self.decay_steps < 5:
                    if self.decay_iterations > 200: #was 500K
                        self.decay_iterations = 200
                else:
                    scheduled_lr = self.scheduler.state_dict()['_last_lr']
                    self.decay_iterations = self.decay_iterations + 0.001/float(scheduled_lr[0])
                    print("adding to exponential decay iteraions ",scheduled_lr,self.decay_iterations)
                print(" ************************************ Scheduler.step() *****************************", )
                self.scheduler.step()
                print(" ************************************")
                print(self.scheduler.state_dict())
                print(" ************************************ Scheduler.step() COMPLETE *********************", )
            else:
                if self.iterations % 100000 == 0:
                    print("Iteration ",self.iterations, " Batch_iteration ", self.iterations*BATCH_SIZE, " of Decay Iterations ", self.decay_iterations)    
            
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                ns = next_state[idx]
                a = self.model(ns)
                Q_new = reward[idx] + self.gamma * torch.max(a)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


    def save(self, filename='model', describe=None):
        # save the model
        # save the data
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        full_filename = os.path.join(model_folder_path, filename+'.train_data')
        data = {}
        data['gamma'] = self.gamma
        data['lr'] = self.lr
        data['decay_iterations'] = self.decay_iterations
        data['iter_growth_val'] = self.iter_growth_val
        data['model'] = str(self.model.state_dict())
        data['scheduler'] = str(self.scheduler.state_dict())
        with open(full_filename, 'w') as outfile:  
            json.dump(data, outfile)
        full_filename = os.path.join(model_folder_path, filename+'.pth')
        self.model.save(full_filename)
        full_filename = os.path.join(model_folder_path, filename+'.describe')
        if describe is not None:
            with open(full_filename, 'w') as outfile:
                for i,col in describe:
                    o = f'{i}\t {col}\n'
                    outfile.write(o)
