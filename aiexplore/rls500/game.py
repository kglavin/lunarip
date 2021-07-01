import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

from pygame.display import mode_ok
import trajectory


pygame.init()
font = pygame.font.Font('arial.ttf', 25)
white = pygame.Color('white')
black = pygame.Color('black')
red = pygame.Color('red')
blue = pygame.Color('blue')
yellow = pygame.Color('yellow')

width, height = 1200, 900
missile_size = 100
incr_angle = 0.02  # radians   little more than 1 degree
incr_angle_large = 0.2 # radians 11.4 degrees
vertical_angle = math.pi/2
horizontal_angle = 0.0

ground = pygame.Rect(0, height-2, width, 2 )

SPEED = 600
MAX_ITERATIONS=3000

class Action(Enum):
    N_NULL = 0,
    V_UP = 1, 
    V_DOWN = 2, 
    A_UP = 3,
    A_DOWN = 4,
    FIRE = 5,
    V_UP10 = 6,
    V_DOWN10 = 7,
    A_UP10 = 8,
    A_DOWN10 = 9


Point = namedtuple('Point', 'x, y')

class BallisticGameAI:
    def __init__(self, w=width, h=height):
        self.w = w
        self.h = h
        self.missile_size = missile_size
        self.aabattery_loc = Point(100, self.h-20)
        self.aabattery =  pygame.Rect(self.aabattery_loc.x, self.aabattery_loc.y, 15, 18 )
        self.score = 0
        self.frame_iteration = 0
        self.shots = 0
        self.last_action = None
        self.want_save = False
        self.want_ui = 0
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Ballistic')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        print("previous: iterations",self.frame_iteration," shots: ", self.shots )
        self.score = 0
        self.frame_iteration = 0
        self.missile_loc = Point(1,1)
        self.missile_alpha = math.atan(self.h-self.missile_loc.y/self.missile_loc.x)
        self.missile_range = int(math.sqrt(self.missile_loc.y**2 + self.missile_loc.x**2))
        self.aabattery_loc = Point(100, self.h-20)
        self.aabattery =  pygame.Rect(self.aabattery_loc.x, self.aabattery_loc.y, 15, 18 )
        self.angle = math.pi / 4 
        self.velocity = 600
        self.dt = 0.005
        self.shots = 100
        self.x = []
        self.y = []
        self.shotfired = False
        self.bullet_paths = []
        self.find_missile()

    def play_step(self, action):
        reward = 0
        game_over = False
        missile_hit = False
        self.frame_iteration += 1

        # process the pygame inputs 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_s]:
            self.want_save = True
        if keys_pressed[pygame.K_d]:
            self.want_ui = 1
        if keys_pressed[pygame.K_c]:
            self.want_ui = 0
        if keys_pressed[pygame.K_UP]:
            self.missile_size += 10
        if keys_pressed[pygame.K_DOWN]:
            self.missile_size -= 10
            if self.missile_size < 10:
                self.missile_size = 10



        reward = self._close_angle_reward(action)
        self._move(action) 
        self.last_action = action

        if self.shots == 0 or self.frame_iteration > MAX_ITERATIONS:
            game_over = True
            reward = reward + self.score*100
            return reward, game_over, self.score
    
        if self.shotfired == True:
            r,missile_hit = self._shot_fired_reward(action)
            reward += r

        #update ui and clock
        if self.want_ui > 0:
            self._update_ui()
        self.clock.tick(SPEED)

        # clean up bullet path
        self.bullet_paths = []

        # add a new missile if needed
        if missile_hit == True:
            self.find_missile()
            self._update_ui()

        return reward, game_over, self.score


    def _update_ui(self):
        # base screen
        pygame.draw.rect(self.display, black, pygame.Rect(0,0, width, height))
        pygame.draw.rect(self.display, yellow, ground)
        pygame.draw.rect(self.display, white, self.aabattery)

        #missile
        if self.missile is not None:
            pygame.draw.rect(self.display, red, self.missile)
            pygame.draw.rect(self.display, white,pygame.Rect(self.missile_loc.x+self.missile_size//2,
                                                            self.missile_loc.y+self.missile_size//2,
                                                            2,2))

        # aiming hint
        x,y =  trajectory.trajectory(4,self.angle,self.velocity,self.dt*10)
        if len(x) > 0:
            for i in range(len(x)):
                p = pygame.Rect(self.aabattery.x +10 +x[i],
                                self.aabattery.y -20 -y[i], 1, 1)
                pygame.draw.rect(self.display, white, p)
            if self.want_ui > 0:
                pygame.time.delay(20)

        #bullet trajectory if present
        for bullet in self.bullet_paths:
            pygame.draw.rect(self.display, blue, bullet)

        # status line
        status = "angle: " + "{:.4f}".format(self.angle) + " velocity: " + str(self.velocity) 
        status = status + " shots: " + str(self.shots) + " score: " + str(self.score) 
        status = status + " iteration: " + str(self.frame_iteration) + " (" + str(self.missile_size) +","+ "{:.4f}".format(self.missile_alpha)+","+str(self.missile_range)+")"
        status = status + " la: " + str(self.last_action)

        text = font.render(status, True, white)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

        # slight delay to allow bullet path to be seen
        if len(self.bullet_paths) > 0:
            if self.want_ui > 0:
                pygame.time.delay(200)

        return

    def _shot_fired_reward(self,action):
        # got a new trajectory because of a fire action, check it it hits the missile 
        self.shots -= 1
        reward = -500 # assume a miss and punish 
        missile_hit = False
        self.shotfired = False
        if len(self.x) > 0:
            for i in range(len(self.x)):
                if i % 3 == 0:
                    bullet = pygame.Rect(self.aabattery.x +10 +self.x[i],
                                self.aabattery.y -20 -self.y[i], 1, 1)
                    self.bullet_paths.append(bullet)
                    if self.missile.colliderect(bullet):
                        self.score +=1
                        reward += 2500
                        missile_hit=True
                        break
        # remove the trajectory 
        self.x = []
        self.y = []
        return reward, missile_hit

    def _close_angle_reward(self,action):
        reward = 0

        if action == Action.A_UP:
            if self.angle > self.missile_alpha:
                reward += -10
            else:
                reward += 0.2
                if self.angle <= (self.missile_alpha - incr_angle_large):
                    reward += 0.25
                if self.angle <= (self.missile_alpha - incr_angle):
                    reward += 2.6

        if action == Action.A_UP10:
            if self.angle > self.missile_alpha:
                reward += -10
            else:
                reward += 0.2
                if self.angle <= (self.missile_alpha - incr_angle_large):
                    reward += 2.6
                if self.angle <= (self.missile_alpha - incr_angle):
                    reward += - 10

        if action == Action.A_DOWN:
            if self.angle < self.missile_alpha:
                reward += -10
            else:
                reward += 0.2
                if self.angle >= (self.missile_alpha + incr_angle_large):
                    reward += 0.25
                if self.angle >= (self.missile_alpha + incr_angle):
                    reward += 2.6

        if action == Action.A_DOWN10:
            if self.angle < self.missile_alpha:
                reward += -10
            else:
                reward += 0.2
                if self.angle >= (self.missile_alpha + incr_angle_large):
                    reward += 2.6
                if self.angle <= (self.missile_alpha + incr_angle):
                    reward += -10
        return reward


    def _move(self, action):
            if action == Action.V_UP and self.velocity <= 200:
                #self.velocity += 1
                pass

            if action == Action.V_UP10 and self.velocity <= 200:
                #self.velocity += 10
                pass

            if action == Action.V_DOWN and self.velocity >= 200:
                #self.velocity -= 1
                pass
            
            if action == Action.V_DOWN10 and self.velocity >= 210:
                #self.velocity -= 10
                pass

            if action == Action.A_UP and self.angle < math.pi/2:
                self.angle += 0.025
                if self.angle > math.pi/2:
                    self.angle = math.pi/2
            
            if action == Action.A_UP10 and self.angle < math.pi/2:
                self.angle += 0.1
                if self.angle > math.pi/2:
                    self.angle = math.pi/2

            if action == Action.A_DOWN and self.angle > 0.1:
                self.angle -= 0.025
                if self.angle <= 0.1:
                    self.angle = 0.1
            
            if action == Action.A_DOWN10 and self.angle > 0.1:
                self.angle -= 0.1
                if self.angle <= 0.1:
                    self.angle = 0.1

            if action == Action.FIRE and self.shots > 0:
                    self.shotfired = True
                    self.x,self.y =  trajectory.trajectory(1000,self.angle,self.velocity,self.dt)
            return

    def find_missile(self):
        self.missile_loc = Point(random.randint(self.w-1100, self.w-60),
                                random.randint(self.h-850, self.h-100)) 
        #self.missile_loc = Point(300+random.randint(-250,850),
        #                        300+random.randint(-50,500))
        #self.missile_loc = Point(700,self.h-200)
        
        #self.missile_alpha = math.atan((self.h - self.missile_loc.y)/self.missile_loc.x)

        self.missile_alpha = math.atan((self.aabattery.y-20-self.missile_loc.y)/(self.missile_loc.x-self.aabattery.x+10))
        self.missile_range = int(math.sqrt(self.missile_loc.y**2 + self.missile_loc.x**2))
        
        self.missile = pygame.Rect(self.missile_loc.x, self.missile_loc.y, self.missile_size, self.missile_size)
        self.angle = random.randint(10,1500)/1000
    

if __name__ == "__main__":

    game=BallisticGameAI()

    al = list(Action)
    while True:
        game.play_step(random.choice(al))
