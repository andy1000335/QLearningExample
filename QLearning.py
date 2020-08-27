import numpy as np
import pandas as pd
import time
import os

COL_NUM = 5
ROW_NUM = 5
ACTION_LIST = ['left', 'right', 'up', 'down']
ACTION_NUM = len(ACTION_LIST)
TARGET_POS = ('x4', 'y4')
TRAP_POS = ('x2', 'y2')
FRESH_TIME = 0.08

EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 30

def creat_table():
    '''
    Table:
          x0             x1             x2             x3             x4
          y0 y1 y2 y3 y4 y0 y1 y2 y3 y4 y0 y1 y2 y3 y4 y0 y1 y2 y3 y4 y0 y1 y2 y3 y4
    left   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    right  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    up     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    down   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    '''
    col = sum([['x'+str(col)]*ROW_NUM for col in range(COL_NUM)], [])
    row = ['y'+str(row) for row in range(ROW_NUM)]*COL_NUM
    val = [[0 for _ in range(COL_NUM*ROW_NUM)]]*ACTION_NUM

    table = pd.DataFrame(val, columns=pd.MultiIndex.from_tuples(zip(col, row)), index=ACTION_LIST)
    # print(table.loc['left', ('x0', 'y0')])    # get value
    
    return table

def choose_action(table, state):
    '''
    state = ('x0', 'y0')
    '''
    stateAction = table.loc[:, state]
    if (np.random.uniform() > EPSILON) or ((stateAction == 0).all()):
        action = np.random.choice(ACTION_LIST)
    else:
        action = stateAction.idxmax()
    return action

def environment(state, target_pos=TARGET_POS, trap_pos=TRAP_POS):
    '''
    state = ('x0', 'y0')
    trap_pos = ('x2', 'y2')
    target_pos = ('x4', 'y4')
    
    O ┬ ┬ ┬ ┐
    ├ ┼ ┼ ┼ ┤
    ├ ┼ # ┼ ┤
    ├ ┼ ┼ ┼ ┤
    └ ┴ ┴ ┴ X
    '''
    os.system('cls')
    stateX = int(state[0][1])
    stateY = int(state[1][1])
    targetX = int(target_pos[0][1])
    targetY = int(target_pos[1][1])
    trapX = int(trap_pos[0][1])
    trapY = int(trap_pos[1][1])
    
    env = ('┌ '+'┬ '*(COL_NUM-2)+'┐,' + ('├ '+'┼ '*(COL_NUM-2)+'┤,')*(ROW_NUM-2) + '└ '+'┴ '*(COL_NUM-2)+'┘').split(',')
    env[targetY] = env[targetY][0: targetX*2] + 'X' + env[targetY][(targetX*2+1): ]
    env[trapY] = env[trapY][0: trapX*2] + '#' + env[trapY][(trapX*2+1): ]
    env[stateY] = env[stateY][0: stateX*2] + 'O' + env[stateY][(stateX*2+1): ]

    for row in env:
        print(row)
    time.sleep(FRESH_TIME)
    
def get_next_state(state, action):
    x = int(state[0][1])
    y = int(state[1][1])
    reword = 0
    if (action == 'left') and (x != 0):
        x -= 1
    elif (action == 'right') and (x != (COL_NUM-1)):
        x += 1
    elif (action == 'up') and (y != 0):
        y -= 1
    elif (action == 'down') and (y != (ROW_NUM-1)):
        y += 1
    state_ = ('x'+str(x), 'y'+str(y))

    if state_ == TARGET_POS:
        reword = 5
    elif state_ == TRAP_POS:
        reword = -1

    return state_, reword

def run_QLearning():
    QTable = creat_table()
    for episode in range(MAX_EPISODES):
        step = 0
        state = ('x0', 'y0')    # initial position
        isTerminal = False
        environment(state)
        while not isTerminal:
            action = choose_action(QTable, state)
            state_, reword = get_next_state(state, action)
            qVal = QTable.loc[action, state]
            if (state_ != TARGET_POS) and (state_ != TRAP_POS):
                qVal_ = reword + GAMMA*QTable.loc[:, state_].max()
            else:
                qVal_ = reword
                isTerminal = True
            QTable.loc[action, state] += ALPHA * (qVal_ - qVal)

            state = state_
            environment(state)
            step += 1
        print('Episode {}: total step = {}'.format(episode+1, step))
        time.sleep(1)
    return QTable

if __name__ == "__main__":
    table = run_QLearning()
