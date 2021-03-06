import pygame
import random


import numpy as np
import math

from pygame.display import mode_ok
import trajectory
from agenttypes import Point, Action,StateStanza
from agenttypes import onehot_action,action_onehot,int_onehot,action_list


pygame.init()
font = pygame.font.Font('arial.ttf', 12)
white = pygame.Color('white')
black = pygame.Color('black')
red = pygame.Color('red')
blue = pygame.Color('blue')
yellow = pygame.Color('yellow')
grey = pygame.Color('grey')

width, height = 1200, 900
width, height = 640, 480
max_range = math.sqrt(width**2+height**2)

missile_size = 40
incr_angle = 0.011  # radians   little more than 1 degree
incr_angle_large = 0.1 # radians 11.4 degrees
vertical_angle = math.pi/2
horizontal_angle = -0.01

ground = pygame.Rect(0, height-2, width, 2 )

SPEED = 600
MAX_ITERATIONS=4000

ANGLE_DOWN_SMALL = round(0.001,3)
ANGLE_DOWN = round(ANGLE_DOWN_SMALL*10,3)

#ANGLE_DOWN = ANGLE_UP + 0.00001
#ANGLE_DOWN_SMALL = ANGLE_UP_SMALL - 0.00001
ANGLE_UP = round(ANGLE_DOWN,3)
ANGLE_UP_SMALL = round(ANGLE_DOWN_SMALL,3)
ANGLE_FIRE_WOBBLE = round(ANGLE_DOWN_SMALL*1.77 ,3)

HIT_REWARD = 500
MOVE_PENALTY = -100
MOVE_REWARD_NORMAL = 10
MOVE_REWARD_LARGE = (MOVE_REWARD_NORMAL * ANGLE_DOWN/ANGLE_DOWN_SMALL)+1


class AABattery:
    def __init__(self, x=20, y=height-20,z=0):
        self.location = Point(x,y,z)
        self.rect = pygame.Rect(self.location.x, self.location.y, 15, 18 )
        self.alpha = round(random.randint(0,1572)/1000,3)
        self.dt = 0.0005
        self.velocity = 2000

    def render(self,display):
        pygame.draw.rect(display, white, self.rect)
        x,y =  trajectory.trajectory(4,self.alpha,self.velocity,self.dt*10)
        if len(x) > 0:
            for i in range(len(x)):
                p = pygame.Rect(self.location.x +10 +x[i],
                                self.location.y -2 -y[i], 1, 1)
                pygame.draw.rect(display, white, p)
    
    def rotate(self,angle):
        self.alpha += angle
        self.alpha = round(self.alpha,3)

        if self.alpha <= 0.0:
            self.alpha = round(0.001,3)
        if self.alpha > math.pi:
            self.alpha = round(math.pi,3)
    
    def adjustVel(self,velocity):
        self.velocity = velocity
    
    def adjustAngle(self,angle):
        self.angle = angle

class Target:
    def __init__(self, x=100, y=100,z=100, size=missile_size):
        self.location = Point(x,y,z)
        self.size = size
        self.target_range = math.sqrt(x**2+y**2)
        self.target_box = pygame.Rect(self.location.x, self.location.y, self.size, self.size)
        #self.target_box = pygame.Rect(self.location.x, self.location.y, 25+self.size*self.target_range//int(0.90*max_range), 25+self.size*self.target_range//int(0.90*max_range))

    def render(self,display):
        #modified_size = 25+self.size*self.target_range//int(0.90*max_range)
        modified_size = self.size
        pygame.draw.rect(display, grey, self.target_box)
        pygame.draw.rect(display, black, pygame.Rect(self.location.x+1,
                                                            self.location.y+1,
                                                            modified_size-2,
                                                            modified_size-2))
        pygame.draw.rect(display, red,pygame.Rect(self.location.x+modified_size//2,
                                                            self.location.y+modified_size//2,
                                                            4,4))




class BallisticGameAI:
    def __init__(self, w=width, h=height, speed=SPEED,iterations=MAX_ITERATIONS):
        self.w = w
        self.h = h
        self.speed = speed
        self.default_target_size = missile_size
        self.aa = AABattery(x=20, y=height-20)
        self.score = 0
        self.max_iterations=iterations
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
        self.score = 0
        self.frame_iteration = 0
        self.target = Target(50,50,self.default_target_size)
        self.target_alpha = math.atan(self.h-self.target.location.y/self.target.location.x)
        self.target_range = int(math.sqrt(self.target.location.y**2 + self.target.location.x**2))
        self.target_velocity = 600
        self.aa = AABattery(x=20, y=height-20)
        self.dt = 0.0005
        self.shots = 500
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
            self.default_target_size += 1
            if self.default_target_size > 300:
                self.default_target_size = 300
        if keys_pressed[pygame.K_DOWN]:
            self.default_target_size -= 1
            if self.default_target_size < 10:
                self.default_target_size = 10
        if keys_pressed[pygame.K_q]:
            return 0, True, self.score, False

        reward = self._close_angle_reward(action)
        self._move(action) 
        self.last_action = action

        if self.shotfired == True:
            r,missile_hit = self._shot_fired_reward(action)
            reward += r
            self.aa.rotate(random.choice([-2*ANGLE_FIRE_WOBBLE,0,2*ANGLE_FIRE_WOBBLE]))

        if self.shots == 0 or self.frame_iteration > self.max_iterations:
            game_over = True
            reward = reward + self.score
            return reward, game_over, self.score,missile_hit
    

        #update ui and clock
        if self.want_ui > 0:
            self.update_ui()
        self.clock.tick(self.speed)

        # clean up bullet path
        self.bullet_paths = []

        return reward, game_over, self.score,missile_hit

    def get_state(self): 
        angle = round(self.aa.alpha-self.target_alpha,3)
        #return np.array(state, dtype=float)
        state = [angle,self.target_range,self.aa.velocity]
        return np.array(state,dtype=float)



    def update_ui(self):
        # base screen
        pygame.draw.rect(self.display, black, pygame.Rect(0,0, width, height))
        pygame.draw.rect(self.display, yellow, ground)
        #pygame.draw.rect(self.display, white, self.aabattery)
        self.aa.render(self.display)

        #missile
        if self.target is not None:
            self.target.render(self.display)
            if self.want_ui > 0:
                pygame.time.delay(20)

        #bullet trajectory if present
        bc = 0
        for bullet in self.bullet_paths:
            bc +=1
            if bc % 2 == 1:
                pygame.draw.rect(self.display, blue, bullet)

        # status line
        status = "angle: " + "{:.3f}".format(self.aa.alpha-self.target_alpha) + " velocity: " + str(self.aa.velocity) +" Range: "+str(self.target_range)
        status = status + " shots: " + str(self.shots) + " score: " + str(self.score) 
        status = status + " iteration: " + str(self.frame_iteration) + " (" + str(self.target.size) +","+ "{:.3f}".format(self.target_alpha)+")"
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
                    bullet = pygame.Rect(self.aa.location.x +10 +self.x[i],
                                self.aa.location.y -2 -self.y[i], 2, 2)
                    self.bullet_paths.append(bullet)
                    bullet_collide_radius = pygame.Rect(self.aa.location.x +10+4 +self.x[i],
                                self.aa.location.y -2-4 -self.y[i], 10, 10)
                    if self.target.target_box.colliderect(bullet_collide_radius):
                        self.score +=1
                        reward += HIT_REWARD
                        missile_hit=True
                        angle = round(self.aa.alpha-self.target_alpha,3)
                        break
        # remove the trajectory 
        self.x = []
        self.y = []
        return reward, missile_hit

    def _close_angle_reward(self,action):
        reward = 0
        angle = round(self.aa.alpha-self.target_alpha,3)
        
        if angle < ANGLE_DOWN_SMALL and angle > -ANGLE_UP_SMALL:
            if action == Action.FIRE:
                reward += 5
            return reward

        if angle >= ANGLE_DOWN*1.5:
            if action == Action.A_DOWN_LARGE:
                reward += MOVE_REWARD_LARGE            
            else:
                reward += MOVE_PENALTY
            return reward
        
        if angle <= ANGLE_DOWN*1.5 and angle >= ANGLE_DOWN_SMALL:
            if action == Action.A_DOWN:
                reward += MOVE_REWARD_NORMAL
            else:
                reward += MOVE_PENALTY
            return reward

        if angle <= -(ANGLE_UP*1.5):
            if action == Action.A_UP_LARGE:
                reward += MOVE_REWARD_LARGE            
            else:
                reward += MOVE_PENALTY
            return reward

        if angle < -(ANGLE_UP_SMALL):
            if action == Action.A_UP:
                reward += MOVE_REWARD_NORMAL
            else:
                reward += MOVE_PENALTY
            return reward
        return reward  


    def _close_angle_reward2(self,action):
        reward = 0
        angle = round(self.aa.alpha-self.target_alpha,3)
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
            if angle <= -ANGLE_UP*1.25:
                reward += MOVE_REWARD_LARGE

        if action == Action.A_DOWN_LARGE:
            if angle > ANGLE_DOWN*1.25:
                reward += MOVE_REWARD_LARGE
            if angle > ANGLE_DOWN:
                reward += MOVE_REWARD_NORMAL
            if angle < -ANGLE_UP_SMALL:
                reward += MOVE_PENALTY

        return reward


    def _move(self, action):
            if action == Action.A_UP:
                if self.aa.alpha < math.pi:
                    self.aa.rotate(ANGLE_UP_SMALL)
                else:
                    self.aa.adjustAngle(round(math.pi,3))

            if action == Action.A_DOWN:
                if self.aa.alpha > ANGLE_DOWN_SMALL:
                    self.aa.rotate(-ANGLE_DOWN_SMALL)
                else:
                    self.aa.adjustAngle(round(0.001,3))

            if action == Action.A_UP_LARGE:
                if self.aa.alpha < math.pi:
                    self.aa.rotate(ANGLE_UP)
                else: 
                    self.aa.adjustAngle(round(math.pi,3))

            if action == Action.A_DOWN_LARGE: 
                if self.aa.alpha > ANGLE_DOWN:
                    self.aa.rotate(-ANGLE_DOWN)
                else:
                    self.aa.adjustAngle(round(0.001,3))

            if action == Action.FIRE:
                    self.shotfired = True
                    self.x,self.y =  trajectory.trajectory(1500,self.aa.alpha,self.aa.velocity,self.dt)
            return

    def find_missile(self):

        #x,y = random.randint(self.w-1130, self.w-80),random.randint(self.h-800, self.h-50)
        #x,y = random.randint(self.w-1130, self.w-900),random.randint(self.h-500, self.h-50)
        x,y = random.randint(self.w-580, self.w-30),random.randint(self.h-450, self.h-50)
        self.target = Target(x,y)
        ay = self.aa.location.y-2-(self.target.location.y+self.target.size//2)
        ax = (self.target.location.x+self.target.size//2)-self.aa.location.x+10
        if ax == 0:
            ax = 0.000001
        self.target_alpha = round(math.atan(ay/ax),3)
        self.target_range = int(math.sqrt(ay**2 + ax**2))

    
    def hint(self):
        angle = round(self.aa.alpha-self.target_alpha,3)
        if angle < ANGLE_DOWN_SMALL and angle > -ANGLE_UP_SMALL:
            return Action.FIRE

        if angle <= ANGLE_DOWN*1.5 and angle >= ANGLE_DOWN_SMALL:
            return Action.A_DOWN  

        if angle >= ANGLE_DOWN*1.5 :
            return Action.A_DOWN_LARGE
        
        if angle < -(ANGLE_UP_SMALL) and angle > -(ANGLE_UP*1.5) :
            return Action.A_UP 

        if angle <= -(ANGLE_UP*1.5):
            return Action.A_UP_LARGE 

        return random.choice([Action.A_UP, Action.A_DOWN, Action.FIRE])


    def fire_data(self,rng):
        firedata = []
        for r in range(1,rng,1):
            #short range fire at -.012 to +0.012
            for a in range(0,4,1):
                    firedata.append(StateStanza([round(a/1000,3),r,self.aa.velocity],
                                        action_onehot[Action.FIRE],
                                        HIT_REWARD,
                                        [round(a/1000,3), r,self.aa.velocity],
                                        False))
                    firedata.append(StateStanza([round(-a/1000,3),r,self.aa.velocity],
                                        action_onehot[Action.FIRE],
                                        HIT_REWARD,
                                        [round(-a/1000,3), r,self.aa.velocity],
                                        False))
        return firedata

    def small_angle_data(self,rng):
        smdata = []
        for r in range(1,rng,1):
            for a in range(5,60,1):
                state = [round(a/1000,3), r,self.aa.velocity]
                action = action_onehot[Action.A_DOWN]
                a -= ANGLE_DOWN_SMALL
                next_state = [round(a/1000,3), r,self.aa.velocity]
                smdata.append(StateStanza(state, action, MOVE_REWARD_NORMAL, next_state, False))
            for a in range(-5,-60,-1):
                state = [round(a/1000,3), r,self.aa.velocity]
                action = action_onehot[Action.A_UP]
                a += ANGLE_UP_SMALL
                next_state = [round(a/1000,3), r,self.aa.velocity]
                smdata.append(StateStanza(state, action, MOVE_REWARD_NORMAL, next_state, False))

        return smdata


    def synthetic_data(self):
        rng = 1400
        rng = 500
        synthetic_data = self.fire_data(rng)

        for r in range(701,rng,1):
            #longer range narrow down fire at -.006 to +0.006
            for a in range(0,4,1):
                for c in range(0,1,1):
                    synthetic_data.append(StateStanza([round(a/1000,3),r,self.aa.velocity],
                                        action_onehot[Action.FIRE],
                                        HIT_REWARD,
                                        [round(a/1000,3), r,self.aa.velocity],
                                        False))
                    synthetic_data.append(StateStanza([round(-a/1000,3),r,self.aa.velocity],
                                        action_onehot[Action.FIRE],
                                        HIT_REWARD,
                                        [round(-a/1000,3), r,self.aa.velocity],
                                        False))         
        for r in range(1,rng,1):
            for a in range(5,60,1):
                state = [round(a/1000,3), r,self.aa.velocity]
                action = action_onehot[Action.A_DOWN]
                a -= ANGLE_DOWN_SMALL
                next_state = [round(a/1000,3), r,self.aa.velocity]
                synthetic_data.append(StateStanza(state, action, MOVE_REWARD_NORMAL, next_state, False))
            for a in range(-5,-60,-1):
                state = [round(a/1000,3), r,self.aa.velocity]
                action = action_onehot[Action.A_UP]
                a += ANGLE_UP_SMALL
                next_state = [round(a/1000,3), r,self.aa.velocity]
                synthetic_data.append(StateStanza(state, action, MOVE_REWARD_NORMAL, next_state, False))

            for a in range(61,1700,1):
                state = [round(a/1000,3), r,self.aa.velocity]
                action = action_onehot[Action.A_DOWN_LARGE]
                a -= ANGLE_DOWN
                next_state = [round(a/1000,3), r,self.aa.velocity]
                synthetic_data.append(StateStanza(state, action, MOVE_REWARD_LARGE, next_state, False))
            for a in range(-61,-1700,-1):
                state = [round(a/1000,3), r,self.aa.velocity]
                action = action_onehot[Action.A_UP_LARGE]
                a += ANGLE_UP
                next_state = [round(a/1000,3), r,self.aa.velocity]
                synthetic_data.append(StateStanza(state, action, MOVE_REWARD_LARGE, next_state, False))       
        return synthetic_data

    def parse(self,d):
            # takes the state map and creates a normalised key from it 
            # so (a,state)  is input and we want this normalised 
            # to start with a passthrough which returns (a,state) 
            # will work
            if d is not None:
                state, action, reward, next_state, done = d
                return state, action, reward, next_state, done
            return None,None,None,None,None


if __name__ == "__main__":

    game=BallisticGameAI()

    al = list(Action)
    while True:
        game.play_step(random.choice(al))
