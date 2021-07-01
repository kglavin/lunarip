import numpy as np
import matplotlib.pyplot as plt
import math 
import scipy.constants as const

g = -const.g #gravitation constant 


def trajectory(steps,angle, v0,dt): #function that computes trajectory for some launch angle & velocity 
    vx0 = math.cos(angle)*v0  
    vy0 = math.sin(angle)*v0   
 
    vy =  np.zeros(steps) 
    x = np.zeros(steps)  
    y = np.zeros(steps) 
    
    x[0] = 0
    y[0] = vy0*dt
    vy[0] = vy0 + g*dt

    for i in range(1,steps):
        vy[i] = vy[i-1] + (g*dt)
        x[i] = x[i-1] + vx0*dt
        y[i] = y[i-1]+ 0.5*(vy[i-1]+vy[i]) * dt
        if y[i] < y[0]:
            break

    return (x[0:i]//1,y[0:i]//1)
    
    
def plot(t,y):
    fig, ax = plt.subplots()
    ax.plot(t, y)
  

if __name__ == "__main__":
    angle = math.pi / 4 
    v0 = 200
    dt = 0.01
    fig, ax = plt.subplots()
    for a in [ math.pi/3, math.pi / 4, math.pi / 5, math.pi / 6, math.pi / 7, math.pi / 8 ]:
        for v in [ 100, 200,300]:
            x,y =  trajectory(1500,a,v,dt)
            ax.plot(x, y)
    plt.show()
