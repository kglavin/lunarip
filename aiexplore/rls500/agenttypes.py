from enum import Enum
from collections import namedtuple

class Action(Enum):
    A_UP = 0,
    A_DOWN = 1,
    FIRE = 2,
    A_UP_LARGE = 3,
    A_DOWN_LARGE = 4,
    A_Z_UP = 5,
    A_Z_DOWN = 6,
    A_Z_UP_LARGE = 7,
    A_Z_DOWN_LARGE = 8,

onehot_action = { 
    (1,0,0,0,0): Action.A_UP,
    (0,1,0,0,0): Action.A_DOWN,
    (0,0,1,0,0): Action.FIRE,
    (0,0,0,1,0): Action.A_UP_LARGE,
    (0,0,0,0,1): Action.A_DOWN_LARGE,
    
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
            'range',
            'velocity',
            ]

Point = namedtuple('Point', 'x, y,z')
StateStanza = namedtuple('StateStanza', 'state, action, reward, next_state, done')