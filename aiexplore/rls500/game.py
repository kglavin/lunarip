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
missile_size = 38
incr_angle = 0.011  # radians   little more than 1 degree
incr_angle_large = 0.1 # radians 11.4 degrees
vertical_angle = math.pi/2
horizontal_angle = -0.01

ground = pygame.Rect(0, height-2, width, 2 )

SPEED = 600
MAX_ITERATIONS=600

class Action(Enum):
    A_UP = 0,
    A_DOWN = 1,
    FIRE = 2,
    A_UP_LARGE = 3,
    A_DOWN_LARGE = 4

ANGLE_UP_SMALL = math.pi/360
ANGLE_UP = ANGLE_UP_SMALL*20

ANGLE_DOWN = ANGLE_UP + 0.00001
ANGLE_DOWN_SMALL = ANGLE_UP_SMALL - 0.00001

ANGLE_FIRE_WOBBLE = ANGLE_UP_SMALL*0.51

HIT_REWARD = 900
MOVE_PENALTY = -10
MOVE_REWARD_NORMAL = 10
MOVE_REWARD_LARGE = (MOVE_REWARD_NORMAL+1) * ANGLE_UP/ANGLE_UP_SMALL


Point = namedtuple('Point', 'x, y')

class BallisticGameAI:
    def __init__(self, w=width, h=height, speed=SPEED):
        self.w = w
        self.h = h
        self.speed = speed
        self.missile_size = missile_size
        self.aabattery_loc = Point(20, self.h-10)
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
        #print("previous: iterations",self.frame_iteration," shots: ", self.shots )
        self.score = 0
        self.frame_iteration = 0
        self.missile_loc = Point(1,1)
        self.missile_alpha = math.atan(self.h-self.missile_loc.y/self.missile_loc.x)
        self.missile_range = int(math.sqrt(self.missile_loc.y**2 + self.missile_loc.x**2))
        #self.aabattery_loc = Point(1, self.h-20)
        self.aabattery =  pygame.Rect(self.aabattery_loc.x, self.aabattery_loc.y, 15, 18 )
        self.angle = random.randint(10,1500)/1000
        self.velocity = 600
        self.dt = 0.005
        self.shots = 300
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
            self.missile_size += 1
            if self.missile_size > 300:
                self.missile_size = 300
        if keys_pressed[pygame.K_DOWN]:
            self.missile_size -= 1
            if self.missile_size < 10:
                self.missile_size = 10

        reward = self._close_angle_reward(action)
        self._move(action) 
        self.last_action = action

        if self.shots == 0 or self.frame_iteration > MAX_ITERATIONS:
            game_over = True
            reward = reward + self.score
            return reward, game_over, self.score,missile_hit
    
        if self.shotfired == True:
            r,missile_hit = self._shot_fired_reward(action)
            reward += r
            #self.angle += random.choice([2*-ANGLE_FIRE_WOBBLE,-ANGLE_FIRE_WOBBLE,ANGLE_FIRE_WOBBLE,2*ANGLE_FIRE_WOBBLE])
            self.angle += random.choice([-ANGLE_FIRE_WOBBLE,ANGLE_FIRE_WOBBLE])

        #update ui and clock
        if self.want_ui > 0:
            self.update_ui()
        self.clock.tick(self.speed)

        # clean up bullet path
        self.bullet_paths = []

        return reward, game_over, self.score,missile_hit


    def update_ui(self):
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

        # aiming sight for visual inspection
        x,y =  trajectory.trajectory(4,self.angle,self.velocity,self.dt*10)
        if len(x) > 0:
            for i in range(len(x)):
                p = pygame.Rect(self.aabattery.x +10 +x[i],
                                self.aabattery.y -2 -y[i], 1, 1)
                pygame.draw.rect(self.display, white, p)
            if self.want_ui > 0:
                pygame.time.delay(20)

        #bullet trajectory if present
        for bullet in self.bullet_paths:
            pygame.draw.rect(self.display, blue, bullet)

        # status line
        status = "angle: " + "{:.3f}".format(self.angle) + " velocity: " + str(self.velocity) 
        status = status + " shots: " + str(self.shots) + " score: " + str(self.score) 
        status = status + " iteration: " + str(self.frame_iteration) + " (" + str(self.missile_size) +","+ "{:.3f}".format(self.missile_alpha)+","+str(self.missile_range)+")"
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
        reward = -100 # assume a miss and punish 
        missile_hit = False
        self.shotfired = False
        if len(self.x) > 0:
            for i in range(len(self.x)):
                if i % 1 == 0:
                    bullet = pygame.Rect(self.aabattery.x +10 +self.x[i],
                                self.aabattery.y -2 -self.y[i], 2, 2)
                    self.bullet_paths.append(bullet)
                    bullet_collide_radius = pygame.Rect(self.aabattery.x +10+4 +self.x[i],
                                self.aabattery.y -2-4 -self.y[i], 2, 2)
                    if self.missile.colliderect(bullet_collide_radius):
                        self.score +=1
                        reward += HIT_REWARD
                        missile_hit=True
                        angle = round(self.angle-self.missile_alpha,3)
                        break
        # remove the trajectory 
        self.x = []
        self.y = []
        return reward, missile_hit

    def _close_angle_reward(self,action):
        reward = 0
        angle = round(self.angle-self.missile_alpha,3)
        if action == Action.A_UP:
            if angle > ANGLE_DOWN_SMALL:
                reward += MOVE_PENALTY
            if angle < -ANGLE_UP:
                reward += MOVE_REWARD_NORMAL
            if angle < -ANGLE_UP_SMALL:
                reward += MOVE_REWARD_LARGE

        if action == Action.A_DOWN:
            if angle > ANGLE_DOWN_SMALL:
                reward += MOVE_REWARD_LARGE
            if angle > ANGLE_DOWN:
                reward += MOVE_REWARD_NORMAL
            if angle < -ANGLE_UP:
                reward += MOVE_PENALTY

        if action == Action.A_UP_LARGE:
            if angle > ANGLE_DOWN_SMALL:
                reward += MOVE_PENALTY
            if angle < -ANGLE_UP:
                reward += MOVE_REWARD_NORMAL
            if angle <= -ANGLE_UP*2:
                reward += MOVE_REWARD_LARGE

        if action == Action.A_DOWN_LARGE:
            if angle > ANGLE_DOWN*2:
                reward += MOVE_REWARD_LARGE
            if angle > ANGLE_DOWN:
                reward += MOVE_REWARD_NORMAL
            if angle < -ANGLE_UP_SMALL:
                reward += MOVE_PENALTY

        return reward


    def _move(self, action):
            if action == Action.A_UP:
                if self.angle < math.pi/2:
                    self.angle += ANGLE_UP_SMALL
                    if self.angle > math.pi/2:
                        self.angle = math.pi/2
                else:
                    self.angle = math.pi/2

            if action == Action.A_DOWN:
                if self.angle > ANGLE_DOWN_SMALL:
                    self.angle -= ANGLE_DOWN_SMALL
                    if self.angle <= 0.0:
                        self.angle = 0
                else:
                    self.angle = 0

            if action == Action.A_UP_LARGE:
                if self.angle < math.pi/2:
                    self.angle += ANGLE_UP
                    if self.angle > math.pi/2:
                        self.angle = math.pi/2
                else: 
                    self.angle = math.pi/2

            if action == Action.A_DOWN_LARGE: 
                if self.angle > ANGLE_DOWN:
                    self.angle -= ANGLE_DOWN
                    if self.angle <= 0.0:
                        self.angle = 0.001
                else:
                    self.angle = 0.001

            if action == Action.FIRE:
                    self.shotfired = True
                    self.x,self.y =  trajectory.trajectory(1000,self.angle,self.velocity,self.dt)
            return

    def find_missile(self):

        self.missile_loc = Point(random.randint(self.w-1130, self.w-30),
                                random.randint(self.h-900, self.h-50)) 
        #self.missile_loc = Point(random.randint(self.w-900, self.w-30),
        #                        random.randint(self.h-900, self.h-200)) 
        ay = self.aabattery.y-2-(self.missile_loc.y+self.missile_size//2)
        ax = (self.missile_loc.x+self.missile_size//2)-self.aabattery.x+10
        if ax == 0:
            ax = 1
        self.missile_alpha = math.atan(ay/ax)
        self.missile_range = int(math.sqrt(ay**2 + ax**2))
        self.missile = pygame.Rect(self.missile_loc.x, self.missile_loc.y, self.missile_size, self.missile_size)
        self.angle = random.randint(0,15708)/10000
    
    def hint(self):
        angle = round(self.angle-self.missile_alpha,3)
        if angle < ANGLE_DOWN_SMALL and angle > -ANGLE_UP_SMALL:
            return Action.FIRE

        if angle >= ANGLE_DOWN:
            return Action.A_DOWN_LARGE
        
        if angle <= ANGLE_DOWN and angle > ANGLE_DOWN_SMALL:
            return Action.A_DOWN  

        if angle <= -(ANGLE_UP):
            return Action.A_UP_LARGE 

        if angle <= -ANGLE_UP_SMALL:
            return Action.A_UP    

        return random.choice([Action.A_UP, Action.A_DOWN, Action.FIRE])

if __name__ == "__main__":

    game=BallisticGameAI()

    al = list(Action)
    while True:
        game.play_step(random.choice(al))
